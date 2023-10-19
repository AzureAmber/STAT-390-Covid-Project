source('Preprocessing Code/adv_preprocessing.R')

# running average missingness for a column of each country 
x <- data_cur %>%
  select(date, location, excess_mortality) %>%
  # filter(date >= as.Date("2021-01-01")) %>%
  group_by(location) %>%
  mutate(
    mi = ifelse(is.na(excess_mortality), 0, 1),
    miavg = cummean(mi)
  )

ggplot(x, aes(date, miavg)) +
  geom_point(aes(color = location)) +
  scale_x_date(date_breaks = "4 months") +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  theme(axis.text.x = element_text(angle = 90))

y = x %>% filter(between(date, as.Date("2021-09-01"), as.Date("2022-05-01")), miavg < 0.6)
noquote(y$location %>% unique())
View(y %>% group_by(location) %>% summarise(v = max(miavg)))


# tests_units               Jan 2021    Apr 2022
# Algeria   China     Egypt     Haiti     Indonesia Lebanon

# icu_patients              Jan 2021    Apr 2022
# Brazil       China        Colombia     Ecuador      Egypt        Ethiopia     Ghana       
# Haiti        India        Indonesia    Iran         Japan        Kenya        Lebanon     
# Mexico       Morocco      Pakistan     Philippines  Russia       Saudi Arabia Sri Lanka   
# Turkey 

# hosp_patients             Feb 2021    June 2022
# Algeria      Argentina    Brazil       China        Colombia     Ecuador      Egypt       
# Ethiopia     Germany      Ghana        Haiti        India        Indonesia    Iran        
# Japan        Kenya        Lebanon      Mexico       Morocco      Pakistan     Philippines 
# Russia       Saudi Arabia South Korea  Sri Lanka    Turkey  

# weekly_icu_admissions     OMIT FROM DATA

# total_tests               Nov 2020    Apr 2022
# Algeria   Brazil    China     Egypt     Germany   Haiti     Indonesia Iran      Kenya    
# Lebanon

# new_tests                 Nov 2020    Apr 2022
# Algeria   Brazil    China     Egypt     Germany   Ghana     Haiti     Indonesia Iran     
# Kenya     Lebanon   Russia

# positive_rate             Jan 2021    Apr 2022
# Algeria   Brazil    Egypt     Germany   Haiti     Indonesia Lebanon

# tests_per_case            Jan 2021    Apr 2022
# Algeria   Brazil    Egypt     Germany   Haiti     Indonesia Lebanon

# total_vaccinations        Jan 2021    Mar 2022
# Algeria     Egypt       Ethiopia    Ghana       Haiti       Iran        Kenya      
# Morocco     Pakistan    Philippines

# people_vaccinated         Jan 2021    Mar 2022
# Algeria      China        Colombia     Egypt        Ethiopia     Ghana        Haiti       
# Iran         Kenya        Morocco      Pakistan     Philippines  Saudi Arabia

# total_boosters            OMIT FROM DATA

# new_vaccinations          Jul 2021    June 2022
# Algeria      Colombia     Egypt        Ethiopia     Ghana        Haiti        Iran        
# Kenya        Morocco      Pakistan     Philippines  South Africa

# handwashing_facilities    OMIT FROM DATA

# excess_mortality          OMIT FROM DATA

# Generally, data from Jan 2021 to Mar/Apr 2022







