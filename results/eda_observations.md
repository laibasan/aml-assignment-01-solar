# Task 2 exploratory observations

Numbers and comments come from `data/plant1_hourly.csv` (804 hourly rows, 15 May–17 June 2020) and the four figures in `results/figures/`.

## 2.1 AC power vs irradiation

Physics says plant AC output should rise almost linearly with sunlight on the array, and sit near zero at night.

The scatter in `01_ac_vs_irradiation.png` is a tight rising line (Pearson correlation 0.997). Hourly AC is near 0 at 0 kW/m² and reaches about 27,000 kW near 1 kW/m². A few high-irradiation points sit below the main line, so the relationship is strong but not perfect.

## 2.2 Module temperature vs ambient temperature

Physics says module temperature should track air temperature and sit well above it when irradiation is high, because the panels absorb sunlight.

`02_module_vs_ambient.png` shows two regimes: at low irradiation (purple) modules stay near 20–28 °C, close to ambient; at high irradiation (yellow) modules reach about 55–62 °C while ambient is only about 28–34 °C. Correlation between the two temperatures is 0.859, but colour makes clear that the extra module heating is sunlight, not air temperature alone.

## 2.3 AC power vs DC power

Physics says AC should be a little smaller than DC (inverter losses), so the ratio AC/DC should be slightly below 1, not a tenth.

`03_ac_vs_dc.png` is an almost perfect straight line, but daytime AC/DC has mean 0.0977 (min 0.0965, max 0.0988). That is a conversion efficiency of ~10%, which is not a plausible inverter. The Plant 1 generation file almost certainly records DC_POWER on a different scale (about 10× AC). AC_POWER is still internally consistent, so it remains the regression target; DC is not used as a feature.

## 2.4 Average AC power by hour of day

Physics says a clear-sky plant in India should be near zero at night, rise after sunrise, peak around solar noon, and fall through the afternoon.

`04_avg_ac_by_hour.png` matches that shape: hours 0–5 and 19–23 average about 0 kW, output rises from 06:00, peaks at 11:00 (mean 21,028 kW), stays high at 12:00, then falls through 18:00. The 11:00 peak is consistent with local solar noon in this time zone.
