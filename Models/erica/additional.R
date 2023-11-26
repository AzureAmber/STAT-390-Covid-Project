# 6. Results
prophetsingle_autoplot <- autoplot(prophet_tuned, metric = "rmse")
prophet_single_best <- show_best(prophet_tuned, metric = "rmse")

#save autoplot
jpeg("Models/erica/results/prophet_single/prophet_single_autoplot.jpeg", width = 8, height = 6, units = "in", res = 300)
print(prophetsingle_autoplot)
dev.off()

# 7. Fit Best Model


# changepoint_num = 25, changepoint_range = 0.9
# prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001

prophet_model <- prophet_reg(
  growth = "linear", 
  season = "additive",
  seasonality_yearly = FALSE, 
  seasonality_weekly = FALSE, 
  seasonality_daily = TRUE,
  changepoint_num = 100, 
  changepoint_range = 0.9,
  prior_scale_changepoints = 0.001,
  prior_scale_seasonality = 0.316, 
  prior_scale_holidays = 0.001) %>%
  set_engine('prophet')

prophet_recipe <- recipe(new_cases ~ date, data = train_prophet_us) %>%
  step_dummy(all_nominal_predictors())

prophet_wflow_tuned <- workflow() %>%
  add_model(prophet_model) %>%
  add_recipe(prophet_recipe)

prophet_fit <- fit(prophet_wflow_tuned, data = train_prophet_us)

final_train <- predict(prophet_fit, new_data = train_prophet_update) %>% 
  bind_cols(train_prophet_update) %>% 
  rename(pred = .pred)

library(ModelMetrics)
result_train <- final_train %>%
  group_by(location) %>%
  summarize(rmse_pred_train = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

final_test <- test_prophet %>%
  bind_cols(predict(prophet_fit, new_data = test_prophet)) %>%
  rename(pred = .pred)

result_test <- final_test %>%
  group_by(location) %>%
  summarize(rmse_pred_test = ModelMetrics::rmse(new_cases, pred)) %>%
  arrange(location)

results <- result_train %>% 
  inner_join(result_test, by = "location", suffix = c("rmse_train_pred", "rmse_test_pred"))

write.csv(results, "Results/erica/prophet_single/prophet_single_rmse_results.csv", row.names = FALSE)


## Training Visualization

train_plot <- final_train %>% 
  filter(location == "United States") %>% 
  ggplot(aes(x=date))+
  geom_line(aes(y = new_cases, color = "Actual New Cases"))+
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed")+
  scale_y_continuous(n.breaks = 15)+
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y")+
  theme_minimal()+
  labs(x = "Date",
       y = "New Cases",
       title = "Training: Actual vs. Predicted New Cases in United States in 2023",
       subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.9,
       prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001)",
       caption = "Prophet Univariate",
       color = "")+
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(train_plot, file = "Results/erica/prophet_single/United States_train_pred.jpeg", 
       width=8, height =7, dpi = 300)

#testing visualization
test_plot <- final_test %>% 
  filter(location == "United States") %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y = new_cases, color = "Actual New Cases")) +
  geom_line(aes(y = pred, color = "Predicted New Cases"), linetype = "dashed") +
  scale_y_continuous(n.breaks = 15) + 
  scale_x_date(date_breaks = "3 months", date_labels = "%b %y") +
  theme_minimal() + 
  labs(x = "Date", 
       y = "New Cases", 
       title = "Testing: Actual vs. Predicted New Cases in Argentina in 2023",
       subtitle = "prophet_reg(changepoint_num = 25, changepoint_range = 0.9,
       prior_scale_changepoints = 0.001, prior_scale_seasonality = 0.316, prior_scale_holidays = 0.001)",
       caption = "Prophet Univariate",
       color = "") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(face = "italic", hjust = 0.5),
        legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  scale_color_manual(values = c("Actual New Cases" = "red", "Predicted New Cases" = "blue"))

ggsave(test_plot, file = "Results/erica/prophet_single/Argentina_test_pred.jpeg",
       width=8, height =7, dpi = 300)


