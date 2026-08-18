## =========================================================================
## Simple CNN-LSTM Hybrid, built from scratch in base R
## No tidyr, no keras3, no torch - just base R + stats.
##
## Data: nse_indexes.csv  (Date, Index, Open, High, Low, Close, Volume, Currency)
## Target: next-day realized volatility of NIFTY 50 (or any index you pick)
##
## Architecture:
##   Conv1D (trainable filters, ReLU)  ->  LSTM (single layer)  ->  Dense (linear)
## Trained with plain stochastic gradient descent + manual backprop-through-time.
##
## NOTE ON SCOPE: this is a compact, educational implementation, not a
## production-grade framework. It's meant to demonstrate the CNN-LSTM
## mechanics end-to-end without any deep learning library. For a real
## project deliverable, expect slower training and rougher optimization
## than Keras/Torch would give you - reduce epochs / data size if it's
## too slow on your machine.
## =========================================================================

set.seed(42)

## ---- 1. Load data (base R only) ----
data_path <- "nse_indexes.csv"     # <-- adjust path if needed
target_index <- "NIFTY 50"         # <-- change to any index in the Index column

raw <- read.csv(data_path, stringsAsFactors = FALSE)

df <- raw[raw$Index == target_index, c("Date", "Close")]
df$Date <- as.Date(df$Date)
df <- df[order(df$Date), ]
df <- df[!duplicated(df$Date), ]
df <- df[!is.na(df$Close) & df$Close > 0, ]
rownames(df) <- NULL

cat("Rows loaded for", target_index, ":", nrow(df), "\n")
cat("Date range:", as.character(min(df$Date)), "to", as.character(max(df$Date)), "\n")

## ---- 2. Log returns + realized volatility target ----
n <- nrow(df)
df$log_return <- c(NA, diff(log(df$Close)))
df <- df[-1, ]  # drop first NA row
rownames(df) <- NULL

## Rolling 5-day std dev of returns as the realized volatility proxy
roll_window <- 5
roll_sd <- function(x, w) {
  out <- rep(NA, length(x))
  for (i in w:length(x)) {
    out[i] <- sd(x[(i - w + 1):i])
  }
  out
}
df$realized_vol <- roll_sd(df$log_return, roll_window)
df <- df[!is.na(df$realized_vol), ]
rownames(df) <- NULL

cat("Rows after feature engineering:", nrow(df), "\n")

## ---- 3. Chronological train/test split ----
split_frac <- 0.8
split_idx <- floor(nrow(df) * split_frac)

## ---- 4. Scale using train stats only (no leakage) ----
train_mean <- mean(df$realized_vol[1:split_idx])
train_sd   <- sd(df$realized_vol[1:split_idx])

scale_vol   <- function(x) (x - train_mean) / train_sd
unscale_vol <- function(x) x * train_sd + train_mean

df$vol_scaled <- scale_vol(df$realized_vol)

## ---- 5. Build sliding-window sequences ----
window_size <- 20   # past 20 days -> predict next-day volatility

make_sequences <- function(vec, window_size) {
  n_seq <- length(vec) - window_size
  X <- matrix(0, nrow = n_seq, ncol = window_size)
  y <- numeric(n_seq)
  for (i in 1:n_seq) {
    X[i, ] <- vec[i:(i + window_size - 1)]
    y[i]   <- vec[i + window_size]
  }
  list(X = X, y = y)
}

seqs <- make_sequences(df$vol_scaled, window_size)
target_positions <- (1:length(seqs$y)) + window_size   # index into df for each target

train_mask <- target_positions <= split_idx
test_mask  <- target_positions > split_idx

X_train <- seqs$X[train_mask, , drop = FALSE]
y_train <- seqs$y[train_mask]
X_test  <- seqs$X[test_mask, , drop = FALSE]
y_test  <- seqs$y[test_mask]

cat("Train sequences:", nrow(X_train), "| Test sequences:", nrow(X_test), "\n")

## =========================================================================
## 6. MODEL DEFINITION (Conv1D -> LSTM -> Dense), all from scratch
## =========================================================================

## --- hyperparameters ---
K <- 3     # conv kernel size
F <- 4     # number of conv filters
H <- 8     # LSTM hidden units
Tc <- window_size - K + 1   # conv output length ("valid" padding)

## --- helper functions ---
sigmoid <- function(x) 1 / (1 + exp(-x))

## --- weight initialization (small random values) ---
init_params <- function(K, F, H) {
  scale_c <- sqrt(1 / K)
  scale_l <- sqrt(1 / (F + H))
  list(
    Wc = matrix(rnorm(F * K, sd = scale_c), nrow = F, ncol = K),
    bc = rep(0, F),
    
    Wf_x = matrix(rnorm(H * F, sd = scale_l), H, F), Wf_h = matrix(rnorm(H * H, sd = scale_l), H, H), bf = rep(0, H),
    Wi_x = matrix(rnorm(H * F, sd = scale_l), H, F), Wi_h = matrix(rnorm(H * H, sd = scale_l), H, H), bi = rep(0, H),
    Wo_x = matrix(rnorm(H * F, sd = scale_l), H, F), Wo_h = matrix(rnorm(H * H, sd = scale_l), H, H), bo = rep(0, H),
    Wg_x = matrix(rnorm(H * F, sd = scale_l), H, F), Wg_h = matrix(rnorm(H * H, sd = scale_l), H, H), bg = rep(0, H),
    
    Wy = matrix(rnorm(H, sd = sqrt(1 / H)), nrow = 1, ncol = H),
    by = 0
  )
}

params <- init_params(K, F, H)

## --- forward pass for ONE sample; returns prediction + full cache for backprop ---
forward_pass <- function(x, p) {
  ## ---- Conv1D + ReLU ----
  conv_preact <- matrix(0, nrow = F, ncol = Tc)  # pre-activation (for ReLU derivative)
  conv_out    <- matrix(0, nrow = F, ncol = Tc)  # post-ReLU, feeds into LSTM
  for (f in 1:F) {
    for (t in 1:Tc) {
      window_vals <- x[t:(t + K - 1)]
      conv_preact[f, t] <- sum(p$Wc[f, ] * window_vals) + p$bc[f]
      conv_out[f, t] <- max(0, conv_preact[f, t])   # ReLU
    }
  }
  
  ## ---- LSTM forward over Tc timesteps ----
  h <- vector("list", Tc); c <- vector("list", Tc)
  f_gate <- vector("list", Tc); i_gate <- vector("list", Tc)
  o_gate <- vector("list", Tc); g_gate <- vector("list", Tc)
  x_lstm <- vector("list", Tc)
  
  h_prev <- rep(0, H); c_prev <- rep(0, H)
  
  for (t in 1:Tc) {
    xt <- conv_out[, t]              # F-dim input at this timestep
    x_lstm[[t]] <- xt
    
    ft <- sigmoid(p$Wf_x %*% xt + p$Wf_h %*% h_prev + p$bf)
    it <- sigmoid(p$Wi_x %*% xt + p$Wi_h %*% h_prev + p$bi)
    ot <- sigmoid(p$Wo_x %*% xt + p$Wo_h %*% h_prev + p$bo)
    gt <- tanh(p$Wg_x %*% xt + p$Wg_h %*% h_prev + p$bg)
    
    ct <- ft * c_prev + it * gt
    ht <- ot * tanh(ct)
    
    f_gate[[t]] <- ft; i_gate[[t]] <- it; o_gate[[t]] <- ot; g_gate[[t]] <- gt
    c[[t]] <- ct; h[[t]] <- ht
    
    h_prev <- ht; c_prev <- ct
  }
  
  ## ---- Dense output layer ----
  y_hat <- as.numeric(p$Wy %*% h[[Tc]] + p$by)
  
  cache <- list(x = x, conv_preact = conv_preact, conv_out = conv_out,
                h = h, c = c, f_gate = f_gate, i_gate = i_gate,
                o_gate = o_gate, g_gate = g_gate, x_lstm = x_lstm, y_hat = y_hat)
  list(y_hat = y_hat, cache = cache)
}

## --- backward pass (BPTT) for ONE sample; returns gradients ---
backward_pass <- function(y_true, cache, p) {
  y_hat <- cache$y_hat
  dy <- (y_hat - y_true)   # d(0.5*(y-yhat)^2)/dyhat
  
  grads <- list(
    Wc = matrix(0, F, K), bc = rep(0, F),
    Wf_x = matrix(0, H, F), Wf_h = matrix(0, H, H), bf = rep(0, H),
    Wi_x = matrix(0, H, F), Wi_h = matrix(0, H, H), bi = rep(0, H),
    Wo_x = matrix(0, H, F), Wo_h = matrix(0, H, H), bo = rep(0, H),
    Wg_x = matrix(0, H, F), Wg_h = matrix(0, H, H), bg = rep(0, H),
    Wy = matrix(0, 1, H), by = 0
  )
  
  ## Dense layer gradients
  grads$Wy <- dy * matrix(cache$h[[Tc]], nrow = 1)
  grads$by <- dy
  dh_next <- as.numeric(t(p$Wy)) * dy   # gradient flowing into h at final timestep
  dc_next <- rep(0, H)
  
  d_conv_out <- matrix(0, F, Tc)  # gradient w.r.t. conv1d output, needed for conv backward
  
  for (t in Tc:1) {
    ht <- cache$h[[t]]; ct <- cache$c[[t]]
    ft <- cache$f_gate[[t]]; it <- cache$i_gate[[t]]
    ot <- cache$o_gate[[t]]; gt <- cache$g_gate[[t]]
    xt <- cache$x_lstm[[t]]
    c_prev <- if (t == 1) rep(0, H) else cache$c[[t - 1]]
    h_prev <- if (t == 1) rep(0, H) else cache$h[[t - 1]]
    
    dh <- dh_next
    dc <- dh * ot * (1 - tanh(ct)^2) + dc_next
    
    do_ <- dh * tanh(ct) * ot * (1 - ot)
    df_ <- dc * c_prev * ft * (1 - ft)
    di_ <- dc * gt * it * (1 - it)
    dg_ <- dc * it * (1 - gt^2)
    
    grads$Wf_x <- grads$Wf_x + df_ %*% t(xt); grads$Wf_h <- grads$Wf_h + df_ %*% t(h_prev); grads$bf <- grads$bf + df_
    grads$Wi_x <- grads$Wi_x + di_ %*% t(xt); grads$Wi_h <- grads$Wi_h + di_ %*% t(h_prev); grads$bi <- grads$bi + di_
    grads$Wo_x <- grads$Wo_x + do_ %*% t(xt); grads$Wo_h <- grads$Wo_h + do_ %*% t(h_prev); grads$bo <- grads$bo + do_
    grads$Wg_x <- grads$Wg_x + dg_ %*% t(xt); grads$Wg_h <- grads$Wg_h + dg_ %*% t(h_prev); grads$bg <- grads$bg + dg_
    
    dx_t <- as.numeric(t(p$Wf_x) %*% df_ + t(p$Wi_x) %*% di_ +
                         t(p$Wo_x) %*% do_ + t(p$Wg_x) %*% dg_)
    d_conv_out[, t] <- dx_t
    
    dh_next <- as.numeric(t(p$Wf_h) %*% df_ + t(p$Wi_h) %*% di_ +
                            t(p$Wo_h) %*% do_ + t(p$Wg_h) %*% dg_)
    dc_next <- dc * ft
  }
  
  ## Conv1D backward (through ReLU)
  x <- cache$x
  for (f in 1:F) {
    for (t in 1:Tc) {
      relu_deriv <- as.numeric(cache$conv_preact[f, t] > 0)
      d_pre <- d_conv_out[f, t] * relu_deriv
      window_vals <- x[t:(t + K - 1)]
      grads$Wc[f, ] <- grads$Wc[f, ] + d_pre * window_vals
      grads$bc[f] <- grads$bc[f] + d_pre
    }
  }
  
  grads
}

## --- SGD parameter update ---
sgd_update <- function(p, g, lr) {
  for (nm in names(p)) p[[nm]] <- p[[nm]] - lr * g[[nm]]
  p
}

## =========================================================================
## 7. TRAINING LOOP
## =========================================================================
epochs <- 15
lr <- 0.01
n_train <- nrow(X_train)

cat("\nTraining CNN-LSTM (", epochs, "epochs,", n_train, "samples/epoch )...\n")

for (epoch in 1:epochs) {
  order_idx <- sample(1:n_train)   # shuffle sample order each epoch (not time order within a sample)
  epoch_loss <- 0
  
  for (i in order_idx) {
    x <- X_train[i, ]
    y <- y_train[i]
    
    fwd <- forward_pass(x, params)
    epoch_loss <- epoch_loss + 0.5 * (y - fwd$y_hat)^2
    
    grads <- backward_pass(y, fwd$cache, params)
    params <- sgd_update(params, grads, lr)
  }
  
  cat(sprintf("Epoch %2d/%d - avg train loss: %.6f\n",
              epoch, epochs, epoch_loss / n_train))
}

## =========================================================================
## 8. EVALUATION ON TEST SET
## =========================================================================
n_test <- nrow(X_test)
pred_scaled <- numeric(n_test)

for (i in 1:n_test) {
  fwd <- forward_pass(X_test[i, ], params)
  pred_scaled[i] <- fwd$y_hat
}

pred_vol   <- unscale_vol(pred_scaled)
actual_vol <- unscale_vol(y_test)
pred_vol_clipped <- pmax(pred_vol, 1e-6)

rmse <- sqrt(mean((actual_vol - pred_vol)^2))
mae  <- mean(abs(actual_vol - pred_vol))
qlike <- mean(log(pred_vol_clipped^2) + (actual_vol^2) / (pred_vol_clipped^2))

cat("\n---- CNN-LSTM (from scratch) Test Set Performance ----\n")
cat("RMSE :", round(rmse, 6), "\n")
cat("MAE  :", round(mae, 6), "\n")
cat("QLIKE:", round(qlike, 6), "\n")

## ---- Save predictions for later comparison against GARCH ----
results <- data.frame(
  Date = df$Date[target_positions[test_mask]],
  actual_vol = actual_vol,
  predicted_vol_cnn_lstm = pred_vol
)
write.csv(results, "cnn_lstm_predictions_base_r.csv", row.names = FALSE)
cat("\nPredictions saved to cnn_lstm_predictions_base_r.csv\n")

## ---- Quick base-R plot ----
plot(results$Date, results$actual_vol, type = "l", col = "black",
     xlab = "Date", ylab = "Realized Volatility",
     main = paste("CNN-LSTM (from scratch) Forecast vs Actual -", target_index))
lines(results$Date, results$predicted_vol_cnn_lstm, col = "red")
legend("topright", legend = c("Actual", "CNN-LSTM Forecast"),
       col = c("black", "red"), lty = 1)

