################################################################################
# ARIMA-GARCH Benchmark Model - ALIGNED to match CNN-LSTM script (proj1.R)
#
# For: "CNN-LSTM Hybrid for Stock Price Movement and Volatility Prediction:
#       Benchmarking Against Classical ARIMA and GARCH Models"
#
# Data: nse_indexes.csv, NIFTY 50, full history (matches proj1.R exactly)
# Volatility target: 5-day rolling std of log returns (matches proj1.R exactly)
# Train/test split: 80/20 chronological (matches proj1.R exactly)
################################################################################

# install.packages(c("tseries","forecast","rugarch","ggplot2"))
library(tseries)
library(forecast)
library(rugarch)
library(ggplot2)

set.seed(42)

# ------------------------------------------------------------------------
# 1. LOAD & PREPROCESS -- mirrors proj1.R steps 1-2 exactly
# ------------------------------------------------------------------------
data_path    <- "nse_indexes.csv"
target_index <- "NIFTY 50"

raw <- read.csv(data_path, stringsAsFactors = FALSE)

df <- raw[raw$Index == target_index, c("Date", "Close")]
df$Date <- as.Date(df$Date)
df <- df[order(df$Date), ]
df <- df[!duplicated(df$Date), ]
df <- df[!is.na(df$Close) & df$Close > 0, ]
rownames(df) <- NULL

cat("Rows loaded for", target_index, ":", nrow(df), "\n")
cat("Date range:", as.character(min(df$Date)), "to", as.character(max(df$Date)), "\n")

df$log_return <- c(NA, diff(log(df$Close)))
df <- df[-1, ]
rownames(df) <- NULL

# Same 5-day rolling realized volatility target as proj1.R
roll_window <- 5
roll_sd <- function(x, w) {
  out <- rep(NA, length(x))
  for (i in w:length(x)) out[i] <- sd(x[(i - w + 1):i])
  out
}
df$realized_vol <- roll_sd(df$log_return, roll_window)
df <- df[!is.na(df$realized_vol), ]
rownames(df) <- NULL

cat("Rows after feature engineering:", nrow(df), "\n")

# ------------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT -- identical formula to proj1.R (80/20, same nrow(df))
# ------------------------------------------------------------------------
split_frac <- 0.8
split_idx  <- floor(nrow(df) * split_frac)

train <- df[1:split_idx, ]
test  <- df[(split_idx + 1):nrow(df), ]

cat("Train size:", nrow(train), " | Test size:", nrow(test), "\n")
cat("Train dates:", as.character(min(train$Date)), "to", as.character(max(train$Date)), "\n")
cat("Test dates: ", as.character(min(test$Date)),  "to", as.character(max(test$Date)),  "\n")

train_returns <- train$log_return
test_returns  <- test$log_return
n_test <- nrow(test)

# ------------------------------------------------------------------------
# 3. ARIMA -- price/return direction prediction
# (No equivalent output in proj1.R -- CNN-LSTM there is volatility-only.
#  This is an EXTRA metric for your report, not a head-to-head number.)
# ------------------------------------------------------------------------
arima_model <- auto.arima(train_returns, stepwise = TRUE, approximation = FALSE, trace = TRUE)
cat("\n--- Selected ARIMA model ---\n")
print(summary(arima_model))

arima_forecasts <- numeric(n_test)
history <- train_returns
for (i in 1:n_test) {
  fit_i <- Arima(history, model = arima_model)
  arima_forecasts[i] <- forecast(fit_i, h = 1)$mean[1]
  history <- c(history, test_returns[i])
}

arima_rmse <- sqrt(mean((arima_forecasts - test_returns)^2))
arima_mae  <- mean(abs(arima_forecasts - test_returns))
directional_accuracy <- mean(sign(test_returns) == sign(arima_forecasts)) * 100

cat("\n=== ARIMA Results (Return Prediction) ===\n")
cat("RMSE:", round(arima_rmse, 6), "| MAE:", round(arima_mae, 6),
    "| Directional Accuracy:", round(directional_accuracy, 2), "%\n")

# ------------------------------------------------------------------------
# 4. GARCH -- volatility prediction, evaluated against SAME target as
#    proj1.R (5-day rolling realized_vol), on the SAME test rows.
# ------------------------------------------------------------------------
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)

garch_fit <- ugarchfit(spec = garch_spec, data = train_returns)
print(garch_fit)

# Efficient rolling 1-step-ahead forecast using rugarch's built-in roller.
# Refitting a fresh GARCH model at every single one of the ~1,300 test days
# (a naive for-loop) is extremely slow (1000+ full optimizer runs). ugarchroll
# does the same recursive-window rolling forecast but refits periodically
# (every `refit.every` days) and reuses the fit in between -- this is the
# standard tool for this exact task and finishes in a couple of minutes.
all_returns <- df$log_return   # full series (train + test), same object proj1.R indexes by

roll <- ugarchroll(
  spec          = garch_spec,
  data          = all_returns,
  n.ahead       = 1,
  forecast.length = n_test,
  refit.every   = 25,          # refit every 25 days instead of every single day
  refit.window  = "recursive", # growing window, same behavior as the original for-loop
  solver        = "hybrid",
  calculate.VaR = FALSE,
  keep.coef     = FALSE
)

roll_df <- as.data.frame(roll)
cat("\nColumns returned by ugarchroll:", paste(names(roll_df), collapse = ", "), "\n")
# NOTE: rugarch versions differ slightly in column naming (usually "Sigma").
# If the line below errors, run head(roll_df) and swap in the correct column name.
garch_forecasts <- roll_df$Sigma   # 1-day-ahead forecasted conditional std dev

# Ground truth = same realized_vol column proj1.R used (5-day rolling std)
actual_vol    <- test$realized_vol
predicted_vol <- garch_forecasts
predicted_vol_clipped <- pmax(predicted_vol, 1e-6)

garch_rmse <- sqrt(mean((predicted_vol - actual_vol)^2))
garch_mae  <- mean(abs(predicted_vol - actual_vol))

# QLIKE -- identical formula to proj1.R
qlike <- mean(log(predicted_vol_clipped^2) + (actual_vol^2) / (predicted_vol_clipped^2))

cat("\n=== GARCH Results (Volatility Prediction) ===\n")
cat("RMSE :", round(garch_rmse, 6), "\n")
cat("MAE  :", round(garch_mae, 6), "\n")
cat("QLIKE:", round(qlike, 6), "\n")

# ------------------------------------------------------------------------
# 5. SAVE PREDICTIONS -- same format/column names as proj1.R's output,
#    so you can merge by Date for a combined plot/table.
# ------------------------------------------------------------------------
garch_results <- data.frame(
  Date = test$Date,
  actual_vol = actual_vol,
  predicted_vol_garch = predicted_vol
)
write.csv(garch_results, "garch_predictions.csv", row.names = FALSE)

arima_results <- data.frame(
  Date = test$Date,
  actual_return = test_returns,
  predicted_return_arima = arima_forecasts
)
write.csv(arima_results, "arima_predictions.csv", row.names = FALSE)

summary_table <- data.frame(
  Model = c("ARIMA (Return Prediction)", "GARCH (Volatility Prediction)"),
  RMSE  = c(round(arima_rmse, 6), round(garch_rmse, 6)),
  MAE   = c(round(arima_mae, 6), round(garch_mae, 6)),
  Extra_Metric = c(paste0(round(directional_accuracy, 2), "% directional accuracy"),
                    paste0("QLIKE: ", round(qlike, 6)))
)
write.csv(summary_table, "arima_garch_summary.csv", row.names = FALSE)
cat("\n=== SUMMARY ===\n")
print(summary_table)

# ------------------------------------------------------------------------
# 6. OPTIONAL: MERGE WITH CNN-LSTM PREDICTIONS FOR A 3-WAY COMPARISON PLOT
# Run this section AFTER you have cnn_lstm_predictions_base_r.csv from proj1.R
# ------------------------------------------------------------------------
cnn_path <- "cnn_lstm_predictions_base_r.csv"
if (file.exists(cnn_path)) {
  cnn <- read.csv(cnn_path, stringsAsFactors = FALSE)
  cnn$Date <- as.Date(cnn$Date)

  merged <- merge(garch_results, cnn[, c("Date", "predicted_vol_cnn_lstm")], by = "Date")

  cnn_rmse  <- sqrt(mean((merged$predicted_vol_cnn_lstm - merged$actual_vol)^2))
  cnn_mae   <- mean(abs(merged$predicted_vol_cnn_lstm - merged$actual_vol))
  cnn_qlike <- mean(log(pmax(merged$predicted_vol_cnn_lstm, 1e-6)^2) +
                     (merged$actual_vol^2) / (pmax(merged$predicted_vol_cnn_lstm, 1e-6)^2))

  final_comparison <- data.frame(
    Model = c("GARCH", "CNN-LSTM"),
    RMSE  = c(round(garch_rmse, 6), round(cnn_rmse, 6)),
    MAE   = c(round(garch_mae, 6), round(cnn_mae, 6)),
    QLIKE = c(round(qlike, 6), round(cnn_qlike, 6))
  )
  cat("\n=== FINAL HEAD-TO-HEAD: GARCH vs CNN-LSTM (Volatility) ===\n")
  print(final_comparison)
  write.csv(final_comparison, "final_comparison_garch_vs_cnnlstm.csv", row.names = FALSE)

  ggplot(merged, aes(x = Date)) +
    geom_line(aes(y = actual_vol, color = "Actual Realized Vol")) +
    geom_line(aes(y = predicted_vol_garch, color = "GARCH Forecast")) +
    geom_line(aes(y = predicted_vol_cnn_lstm, color = "CNN-LSTM Forecast")) +
    labs(title = paste(target_index, "- Volatility Forecast Comparison"),
         y = "Volatility (5-day rolling std)", color = "") +
    theme_minimal()
} else {
  cat("\n[Note] cnn_lstm_predictions_base_r.csv not found in working directory yet.\n")
  cat("Run proj1.R first, then re-run this section to get the 3-way comparison.\n")
}
