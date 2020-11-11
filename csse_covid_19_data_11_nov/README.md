# JHU CSSE COVID-19 Dataset

## Table of contents

 * [Daily reports (csse_covid_19_daily_reports)](#daily-reports-csse_covid_19_daily_reports)
 * [USA daily state reports (csse_covid_19_daily_reports_us)](#usa-daily-state-reports-csse_covid_19_daily_reports_us)
 * [Time series summary (csse_covid_19_time_series)](#time-series-summary-csse_covid_19_time_series)
 * [Data modification records](#data-modification-records)
 * [Retrospective reporting of (probable) cases and deaths](#retrospective-reporting-of-probable-cases-and-deaths)
 * [Large-scale back distributions](#large-scale-back-distributions)
 * [Irregular Update Schedules](#irregular-update-schedules)
 * [UID Lookup Table Logic](#uid-lookup-table-logic)
---

## [Daily reports (csse_covid_19_daily_reports)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)

This folder contains daily case reports. All timestamps are in UTC (GMT+0).

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>FIPS</b>: US only. Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Admin2</b>: County name. US only.
* <b>Province_State</b>: Province, state or dependency name.
* <b>Country_Region</b>: Country, region or sovereignty name. The names of locations included on the Website correspond with the official designations used by the U.S. Department of State.
* <b>Last Update</b>: MM/DD/YYYY HH:mm:ss  (24 hour format, in UTC).
* <b>Lat</b> and <b>Long_</b>: Dot locations on the dashboard. All points (except for Australia) shown on the map are based on geographic centroids, and are not representative of a specific address, building or any location at a spatial scale finer than a province/state. Australian dots are located at the centroid of the largest city in each state.
* <b>Confirmed</b>: Counts include confirmed and probable (where reported).
* <b>Deaths</b>: Counts include confirmed and probable (where reported).
* <b>Recovered</b>: Recovered cases are estimates based on local media reports, and state and local reporting when available, and therefore may be substantially lower than the true number. US state-level recovered cases are from [COVID Tracking Project](https://covidtracking.com/).
* <b>Active:</b> Active cases = total cases - total recovered - total deaths.
* <b>Incident_Rate</b>: Incidence Rate = cases per 100,000 persons.
* <b>Case_Fatality_Ratio (%)</b>: Case-Fatality Ratio (%) = Number recorded deaths / Number cases.
* All cases, deaths, and recoveries reported are based on the date of initial report. Exceptions to this are noted in the "Data Modification" and "Retrospective reporting of (probable) cases and deaths" subsections below.  

### Update frequency
* Since June 15, We are moving the update time forward to occur between 04:45 and 05:15 GMT to accommodate daily updates from India's Ministry of Health and Family Welfare.
* Files on and after April 23, once per day between 03:30 and 04:00 UTC. 
* Files from February 2 to April 22: once per day around 23:59 UTC.
* Files on and before February 1: the last updated files before 23:59 UTC. Sources: [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data) and dashboard.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

### Why create this new folder?
1. Unifying all timestamps to UTC, including the file name and the "Last Update" field.
2. Pushing only one file every day.
3. All historic data is archived in [archived_data](https://github.com/CSSEGISandData/COVID-19/tree/master/archived_data).

---
## [USA daily state reports (csse_covid_19_daily_reports_us)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us)

This table contains an aggregation of each USA State level data.

### File naming convention
MM-DD-YYYY.csv in UTC.

### Field description
* <b>Province_State</b> - The name of the State within the USA.
* <b>Country_Region</b> - The name of the Country (US).
* <b>Last_Update</b> - The most recent date the file was pushed.
* <b>Lat</b> - Latitude.
* <b>Long_</b> - Longitude.
* <b>Confirmed</b> - Aggregated case count for the state.
* <b>Deaths</b> - Aggregated death toll for the state.
* <b>Recovered</b> - Aggregated Recovered case count for the state.
* <b>Active</b> - Aggregated confirmed cases that have not been resolved (Active cases = total cases - total recovered - total deaths).
* <b>FIPS</b> - Federal Information Processing Standards code that uniquely identifies counties within the USA.
* <b>Incident_Rate</b> - cases per 100,000 persons.
* <b>Total_Test_Results</b> - Total number of people who have been tested.
* <b>People_Hospitalized</b> - Total number of people hospitalized. (Nullified on Aug 31, see [Issue #3083](https://github.com/CSSEGISandData/COVID-19/issues/3083))
* <b>Case_Fatality_Ratio</b> - Number recorded deaths * 100/ Number confirmed cases.
* <b>UID</b> - Unique Identifier for each row entry. 
* <b>ISO3</b> - Officialy assigned country code identifiers.
* <b>Testing_Rate</b> - Total test results per 100,000 persons. The "total test results" are equal to "Total test results (Positive + Negative)" from [COVID Tracking Project](https://covidtracking.com/).
* <b>Hospitalization_Rate</b> - US Hospitalization Rate (%): = Total number hospitalized / Number cases. The "Total number hospitalized" is the "Hospitalized – Cumulative" count from [COVID Tracking Project](https://covidtracking.com/). The "hospitalization rate" and "Total number hospitalized" is only presented for those states which provide cumulative hospital data. (Nullified on Aug 31, see [Issue #3083](https://github.com/CSSEGISandData/COVID-19/issues/3083))

### Update frequency
* Once per day between 04:45 and 05:15 UTC.

### Data sources
Refer to the [mainpage](https://github.com/CSSEGISandData/COVID-19).

---
## [Time series summary (csse_covid_19_time_series)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

See [here](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/README.md).

---
## Data modification records
This section will contain any modifications to our datasets as well as the reason for the change. If the error results from an issue on our collection of the data, the error will be listed in the errata.csv in the csse_covid19_time_series folder. If the error results due to a change from the source, the change and reasoning will  be listed below.
Generalized Format: 
Date: Location | Change | Files affected | Reason/Other notes | Source
* February 14: Hubei Province, China | Reduction of 108 deaths | Time_series_covid19_confirmed_global.csv | N/A | N/A
* February 13: Hubei Province, China | Inclusion of probable cases (clinical symptoms) from source | Time_series_covid19_confirmed_global.csv | For lab-confirmed cases only (Before Feb 17), please refer to [who_covid_19_situation_reports](https://github.com/CSSEGISandData/COVID-19/tree/master/who_covid_19_situation_reports) | N/A
* February 27: Italy | Source limits testing to at-risk people showing symptoms of COVID-19 | N/A | N/A | [Source](https://apnews.com/6c7e40fbec09858a3b4dbd65fe0f14f5)
* March 1: Diamond Princess| All cases of COVID-19 in repatriated US citizens from the Diamond Princess are grouped together, and their location is currently designated at the ship’s port location off the coast of Japan. These individuals have been assigned to various quarantine locations (in military bases and hospitals) around the US. This grouping is consistent with the CDC. | N/A | N/A| N/A
* April 13: Hainan Province, China | We responded to the error from 3/24 to 4/1 we had incorrect data for Hainan Province.  We had -6 active cases (168 6 168 -6). We applied the correction (168 6 162 0) that was applied on 4/2 for this period (3/24 to 4/1). | daily reports | N/A | N/A
* April 16: France | After communicating with solidarites-sante.gouv.fr, we decided to make these adjustments based on public available information. From April 4 to April 11, only "cas confirmés" are counted as confirmed cases in our dashboard. Starting from April 12, both "cas confirmés" and "cas possibles en ESMS" (probable cases from ESMS) are counted into confirmed cases in our dashboard. ([More details](https://github.com/CSSEGISandData/COVID-19/issues/2094)) | time_series_covid19_confirmed_global.csv | N/A
* April 17: Wuhan Province, China | Increase in death toll from 2579 to 3869 | time_series_deaths_global.csv | N/A | ([Source1](http://www.china.org.cn/china/Off_the_Wire/2020-04/17/content_75943843.htm), [Source2](http://www.nhc.gov.cn/yjb/s7860/202004/51706a79b1af4349b99264420f2cee54.shtml))
* April 21-22:Benton and Franklin, WA | Data were adjusted/added to match the WA DOH report. See [errata](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/Errata.csv) for details.
* April 22: Navajo Nation, US | Cases within the Navajo Nation had been tracked as an independent data source which resulted in double counting of the cases and deaths within Arizona, New Mexico, and Nevada. The US time series files for confirmed from 4/1 and 4/8 and the US time series files for deaths from 3/31 to 4/17 were corrected to remove the double counting. Adjustments were also made for Navajo County, AZ; Cococino County, AZ; Apache County, AZ; San Juan County, NM; McKinley County, NM; Cibola County, NM; Socorrco County, NM; and San Juan County, UT. See errata file for specfic details.
* April 24, New York City, NY | Back distribution of probable deaths, removal of probable deaths as probable cases | time_series_covid19_confirmed_us.csv, time_series_covid19_deaths_us.csv | This change is in line with CDC reporting guidelines. | N/A
* April 26: Australia | Revision of recovered data from 4/20 to 4/26 | time_series_covid19_recovered_global.csv | N/A | N/A 
* April 28: Other | for consistency, we no longer report the hospitalization data as the max of "current - hospitalized" and "cumulative - hospitalized", and instead only report 'cumulative - hospitalized' from [Covid Tracking Project](https://covidtracking.com/). For states that do not provide cumulative hospital counts no hospital data will be shown.
* April 28: Lithuania | Adjustment in reporting standards for confirmed cases. Prior to April 28, confirmed cases =  the number of positive laboratory test results rather than the number of positive individuals. | N/A | ([Source](https://lietuva.lt/wp-content/uploads/2020/04/UPDATE-April-28.pdf)).
* April 30: United Kingdom | Release of deaths in care homes (prior reporting was hospitalized deaths only). All deaths backdistributed and all values changed | time_series_cvoid19_deaths.csv | N/A | ([Source](coronavirus.data.gov.uk))
* May 1: Kosovo and Serbia | Revision of all data for from 4/19-4/30 adjusted due to stale data source | time_series_covid19_confirmed_global.csv, time_series_covid19_deaths_global.csv, time_series_covid19_recovered_global.csv | N/A | N/A
* May 20: United Kingdom | Large reduction in cases | time_series_covid19_confirmed_global.csv | "This is due to historical data revisions across all pillars." | ([Source](https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public), [DHSCgovuk Twitter](https://twitter.com/DHSCgovuk/status/1263159710892638208)).
* May 27: Netherlands | Official source ceases reporting recovered data | time_series_covid19_recovered_global.csv | For consistency, we have nullified the previous recorded recoveries | N/A
* June 2: France | Reduction in confirmed cases due to a change in calculation method. Since June 2, patients who test positive are only counted once. (Baisse des cas confirmés due à un changement de méthode de calcul. Depuis le 2 juin, les patients testés positifs ne sont plus comptés qu’une seule fois.) ([Source](https://dashboard.covid19.data.gouv.fr/vue-d-ensemble?location=FRA))
* June 5: Chile | On June 2nd, Chile’s Ministerio de Salud began reporting national “total active cases” where in the past they had reported national “total recoveries”.  To accommodate this change and to stay consistent with the ministry’s reporting of active cases, from June 2nd forward we are computing recoveries based on the formula “Active Cases = Total Case – Deaths – Recoveries”.  Based on this, the data for Chile will reflects a jump in recoveries on June 2nd. ([Source](https://www.minsal.cl/nuevo-coronavirus-2019-ncov/casos-confirmados-en-chile-covid-19/))
* June 5: Sweden | In an internal audit of the data for Sweden, it has become clear to our team that our reported total of recoveries conflates regional reporting of the number of patients being released from hospitals with country wide recovery data.  As this regional reporting is not universally available and represents only a subset of recoveries, our prior reporting did not accurately represent nationwide recoveries.  To ensure the accuracy of our data, we have chosen to nullify the number of recovered cases in Sweden until the data is released by the national health ministry. We will also be removing recovery data from our historical time series due to this assessment. | time_series_covid19_recovered_global.csv | N/A | N/A
* June 5: Russia/Ukraine | As noted in the disclaimer for the dashboard, the geographic designations in this data have been designed to be consistent with public guidance from the US State Department.  This does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities.  In implementing subnational data for the Russian Federation and the Ukraine, data for the Crimean Peninsula has been apportioned in line with this guidance.  This adjustment explains a difference in national totals for both the Russian Federation and Ukraine relative to alternate reporting. | All files | N/A | N/A
* June 10: Pakistan | Our previous reporting for Pakistan had a single day delay. A recent update corrected this issue but resulted in data for June 7th being lost. We have corrected this issue by adding June 7th manually and pulling all of the Pakistan data back by a single day. | time_series_covid19_confirmed_global.csv, time_series_covid19_deaths_global.csv, time_series_covid19_recovered_global.csv | N/A | N/A
* June 12: St. Louis City, MO | Data for confirmed cases and deaths from March 16 to June 11 were updated to match up with the updated official report at the [City of St. Louis dashboard](https://www.stlouis-mo.gov/covid-19/data/index.cfm). Date of the first case was updated to March 16, and date of the first deaths was updated to March 23.
* June 12, St. Louis County, MO | data for confirmed cases and deaths from March 9 to June 11 were updated to match up with the updated official report at [St. Louis County government site](https://stlcorona.com/resources/covid-19-statistics1/). Date of the first case was remained on March 8, and date of the first deaths was updated to March 20.
* June 12: Massachusetts | Cases from April 15 to June 11 were updated to match official updateded statistics from the [Massachusetts government raw data - County.csv](https://www.mass.gov/info-details/covid-19-response-reporting). This change arose due to to release of historical probable cases by the state. The alteration distributes probable cases and updates some confirmed case counts that were revised by the state. Dukes and Nantucket are still reported together, though County.csv lists them separately.
* June 16th: Oregon | Delay in reporting from Oregon Health Authority resulted in time series for confirmed and deaths not updating for June 14th - updated via official source. | time_series_covid19_confirmed_global.csv, time_series_covid19_deaths_global.csv | Recovered data was not available for this date. | [Source](https://www.oregon.gov/oha/ERD/Pages/Oregon-reports-101-new-confirmed-presumptive-COVID-19-cases-2-new-deaths.aspx).
* June 19th: Belarus | Case data for April 18th and 19th | time_series_covid19_confirmed_global.csv | Initial error was due to a [delay] (https://news.tut.by/society/681391.html) in reporting by the Belarusian health authorities that wasn't properly distributed. | See prior line for source
* June 25: New Jersey | NJ began reporting probable deaths today and the record for the 25th reflects these 1854 deaths not previously reported. | N/A | Additional information can be found in the [transcript](https://nj.gov/governor/news/news/562020/approved/20200625a.shtml) of the state's June 25th coronavirus briefing.
* June 27: France | Internal audit identified issue with calculation of probable cases in nursing homes for France. The French Health Ministry ended public reporting of this number on June 1st - we have since carried that number of probable cases forward.
* June 30: New York | Increase in deaths in New York by 692 | ([NYC gov](https://www1.nyc.gov/site/doh/covid/covid-19-data.page)) We distributed these data back to the time series tables according to [nychealth GitHub](https://github.com/nychealth/coronavirus-data/blob/master/deaths/probable-confirmed-dod.csv).
* July 3rd: United Kingdom | On July 2nd, the United Kingdom revised their case count due to double counting of cases in England that had been tested in multiple facilities. In doing so, they revised their historical time series data for England (available [here](https://coronavirus.data.gov.uk/)). This change resulted in the need to revise our time series for the United Kingdom. As our time series data represents collective cases in England, Scotland, Northern Ireland, and Wales and the change only affected England, we gathered historical from each respective country's national dashboard (available [here](https://public.tableau.com/profile/public.health.wales.health.protection#!/vizhome/RapidCOVID-19virology-Public/Headlinesummary), [here](https://www.arcgis.com/apps/opsdashboard/index.html#/658feae0ab1d432f9fdb53aa082e4130), and [here](https://app.powerbi.com/view?r=eyJrIjoiZGYxNjYzNmUtOTlmZS00ODAxLWE1YTEtMjA0NjZhMzlmN2JmIiwidCI6IjljOWEzMGRlLWQ4ZDctNGFhNC05NjAwLTRiZTc2MjVmZjZjNSIsImMiOjh9)) to completely rewrite the time series data for cases in the United Kingdom.
* July 9: Japan | Update of confirmed cases from Feb 5 to May 27 and deaths from Feb 13 to May 27 | time_series_covid19_confirmed_global.csv, time_series_covid19_deaths_global.csv | N/A | Updated according to the [Japan COVID-19 Coronavirus Tracker](https://covid19japan.com/)
* July 14: United Kingdom | United Kingdom has made frequent revisions to their death data | time_series_covid19_deaths_global.csv | N/A | Death data was downloaded from [this link](https://coronavirus.data.gov.uk/downloads/csv/coronavirus-deaths_latest.csv) and the death totals for the UK from 3/25 to 6/22 in time_series_covid19_deaths_global.csv were updated to match the data in the official report.
* July 18: Puerto Rico | We are now providing the confirmed cases for Puerto Rico at the municipality (Admin1) level. The historic Admin1 data ranging from 5/6 to 7/17 are from [nytimes dataset](https://github.com/nytimes/covid-19-data). Confirmed cases before 5/6 are categorized into Unassigned, Puerto Rico in `time_series_covid19_confirmed_US.csv`. Meanwhile, deaths are all grouped into Unassigned, Puerto Rico in `time_series_covid19_deaths_US.csv`. Daily cases are from [Puerto Rico Departamento de Salud](http://www.salud.gov.pr/Pages/coronavirus.aspx).
* July 20: Uganda | Recovered data includes Ugandans, non Ugandans and refugees while confirmed data contains Ugandans only. This discrepancy results in negative cases being reported in the daily reports | Daily reports | [Source](https://twitter.com/gbkatatumba/status/1285150623692926976)
* July 22: Liechtenstein | Update to all cases and recovered | time_series_covid19_confirmed_global.csv, time_series_covid19_recovered_global.csv | N/A | Updated in line with historical data provided on this [government website](https://www.llv.li/inhalt/118863/amtsstellen/situationsbericht) and within this [pdf](https://www.llv.li/files/ag/aktuelle-fallzahlen.pdf)
* July 22: Iceland | From June 15 to July 20, the government reported antibody cases. We have removed these cases from our time series file | time_series_covid19_confirmed_global.csv | N/A | [Source](https://www.covid.is/data)
* July 28: Kosovo | Overwriting Kosovo data due to stale source. Data was updated from 3/14 to 7/26. | time_series_covid19_confirmed_global.csv, time_series_covid19_deaths_global.csv, time_series_covid19_recovered_global.csv | N/A | Data revised based on reporting from the [Kosovo National Institute of Public Health](https://www.facebook.com/IKSHPK), the [Kosovo Corona Tracker](https://corona-ks.info/?lang=en), and coincident reporting from local news sources: [Koha Ditore](https://www.koha.net/) and [Telegrafi](https://telegrafi.com/).
* August 17: United Kingdom | Government changes definition of death to those occuring within 28 days of a positive test. We have revised the historical death data to match this reporting. | time_series_covid19_deaths_global.csv | The change in definition results in a loss of around 5000 deaths from the official tally. | Data accessed from the [official webpage](https://coronavirus.data.gov.uk/deaths) on August 17 was used to recreate the time series file.
* August 17: Texas | A backlog of laborartory reporting has been identified in the state of Texas which is causing spikes in reporting at the county level (for reference, see the [Aug 16 press release from Dallas County](https://www.dallascounty.org/Assets/uploads/docs/covid-19/press-releases/august/081620-PressRelease-DallasCountyReports5361AdditionalPositiveCOVID-19Cases.pdf) and local reporting (e.g. [KENS5's reporting in San Antonio](https://www.kens5.com/article/news/local/the-texas-department-of-state-health-services-told-3news-that-walgreens-pharmacy-reported-experiencing-a-coding-error-which-they-have-now-corrected/503-ff7a0eb5-9ce9-4127-82a6-8120175a0d67)).  Data is not currently available that would allow for these positive cases to be appropriately back distributed.
* August 25: US Virgin Islands, US | Improper accession of  data resulted in stale cases and deaths for August 22 and 23. These were corrected using the data available [here](https://www.covid19usvi.com/covid19-report).
* August 25: Collin County, Texas | Case data reset to state level data for August 21-25. The source from the Collin County health department has been removed from the public eye. These adaptations are to align with our new source.
* August 27: Sweden | Government's Public Health Agency published an [official release](https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/allman-information-om-testning/felaktiga-provsvar-i-ca-3-700-covid-19-tester) indicating that approximately 3700 of their cases had been improperly identified with a faulty kit that gave false positive results. The agency cleaned the data on [their dashboard](https://experience.arcgis.com/experience/09f821667ce64bf7be6f9f87457ed9aa/page/page_0/) was corrected to remove these cases over time (with slight changes to deaths as well). We have accessed this data and used it to recreate our cases and deaths time series files.
* August 31: New York City, New York | Borough level data for New York City added to the dashboard. Historical cases and deaths backfilled into the time series files. For description of the approach, please see issue #3084.
* September 2: Luxembourg | Government removes non-resident data from official reports. Recovered time series file adjusted to match official reporting, case data is maintained with previous numbers
* September 10: Walker County, Texas | 453 cases removed from case totals | N/A | County has removed cases associated with the Texas Department of Criminal Justice. A historical correction is not available. | N/A
* September 13: Colorado, Texas | Texas Department of Health notifies that Colorado, Texas was subject to data entry error on September 12 that resulted in 545 cases being reported rather than 454. Time series adjusted to correct this mistake. | time_series_covid19_confirmed_US.csv | N/A | N/A
* September 16: Pennsylvania | Pennsylvania released county level data for September 13 after generation of daily reports. We have used [this report](https://www.health.pa.gov/topics/Documents/Diseases%20and%20Conditions/COVID-19%20County%20Data/County%20Case%20Counts_9-13-2020.pdf) to assign county level data. Of note, the cases for Philadelphia appear to be anomalous in the official report (significant drop of cases) so we have chosen to maintain our previously reported number for this location.
* October 5: Missouri | We have noted irregularities with the reporting for Missouri from September 29-October 1 due to changes in reporting by the Missouri Department of Health. We are working to correct our time series data.
* October 13: Rhode Island, US | in collaboration with the Rhode Island Department of Health, we have been able to recreate the county level death time series for Rhode Island. Moving forward, we will be reporting deaths at the county level | time_series_deaths_us.csv & time_series_deaths_global.csv. Rhode Island will be publishing county level cases and deaths once per week. Please see issue #3229 for more details. 
* October 15: Luxembourg | Update for stale data October 8 through 14 | All time series files and daily reports | Updated via [daily report pdfs](https://data.public.lu/fr/datasets/covid-19-rapports-journaliers/#_) from national source
* October 27: Alaska | Add non-resident cases from March 12 to October 26 | Confirmed cases - time series tables for the US and global | NA | [Cases by date reported](https://coronavirus-response-alaska-dhss.hub.arcgis.com/datasets/geographic-distribution-of-all-cases-by-date-reported)
* October 30: Franklin County, VA | Rewrite time series 8/22 to 10/28 with appropriate data | All time series files and us daily reports for 8/22 through 10/28 | Internal mapping error had lead to cases in Franklin City, VA replacing values for Franklin County, VA as well as the source for Franklin City going stale
* October 31: Greece | Edit recoveries August 4 to present | time_series_covid19_recovered_global.csv | Update recovery data using government press releases | [Source](https://eody.gov.gr/category/deltia-typoy/)
* Unassigned, Colorado | Addition of historical "international" entry to unassigned cateogry | time_series_covid19_confirmed_us.csv, time_series_covid19_confirmed_global.csv | Addition of missing cases from an international entry - cases moving forward will include these cases in unassigned | Data used from the csv file hosted [here](https://data-cdphe.opendata.arcgis.com/datasets/222c9d85e93540dba523939cfb718d76_0)
* November 9: Puerto Rico, US | Revision of historical data in line with clarification from the Puerto Rican health department regarding suspected versus probable cases | time_series_covid19_cases_US.csv, time_series_covid19_cases_global.csv | Prior to November 7, Puerto Rico suspected cases were serology test results, which are incongruent with our working definition of probable cases. The breakdown is now indicated and we have used historical reporting to alter our previous reported totals. This change brings reporting for the territory in line with national reporting for the rest of the US. [Source](http://www.salud.gov.pr/Estadisticas-Registros-y-Publicaciones/Pages/COVID-19.aspx)
* November 9: Georgia, US | Revision of data from November 3-8 in line with newly published antigen data, taken from coincident reporting with the state dashboard | All time series files and daily reports | [Source](https://dph.georgia.gov/covid-19-daily-status-report)
* November 9: Kansas, US | Revision of data from September 20 to November 8 to deconflict differences in reporting between state and county sources. Max of source used as ground truth. | All time files and daily reports | [State source](https://www.coronavirus.kdheks.gov/160/COVID-19-in-Kansas), [County source](https://experience.arcgis.com/experience/9a7d44773e4c4a48b3e09e4d8673961b/page/page_18/)
* November 9: Wisconsin, US | Revision of data beginning October 19th to include probable cases. | All time files and daily reports

## Retrospective reporting of (probable) cases and deaths
This section reports instances where large numbers of historical cases or deaths have been reported on a single day. These reports cause anomalous spikes in our time series curves. When available, we liaise with the appropriate health department and distribute the cases or deaths back over the time series. If these are successful, they will be reported in the below section titled "Large Scale Back Distributions". A large proportion of these spikes are due to the release of probable cases or deaths.
Generalized Format: 
Date: Location | Change | Reason/Other notes | Source 
* April 12: France | Spike in cases | Inclusion of "cas possibles en ESMS" (probable cases from ESMS)| ([More details](https://github.com/CSSEGISandData/COVID-19/issues/2094))
* April 21: Finland | Increase in deaths from 98 to 141 | Finnish National Institute for Health and Welfare included deaths in nursing homes in the Helsinki Metropolitan area for the first time. | [Source](https://www.foreigner.fi/articulo/coronavirus/finland-reports-44-increase-in-number-of-coronavirus-deaths/20200421174642005414.html)
* April 23: New York City, New York, US | Reporting of probable deaths | N/A |[Source](https://www.nbcnews.com/health/health-news/live-blog/2020-04-23-coronavirus-news-n1190201/ncrd1190406#blogHeader) | **Back Distributed**
* April 24: Colorado, US | Spike of 121 deaths | Inclusion of probable deaths for first time | [Source](https://www.denverpost.com/2020/04/24/covid-coronavirus-colorado-new-cases-deaths-april-24/)
* April 24: Republic of Ireland | Spike of 189 deaths | Source begins including probable deaths (those with COVID-19 listed as cause of death but no molecular test) | [Source](https://www.irishnews.com/news/republicofirelandnews/2020/04/24/news/republic-s-covid-19-death-toll-passes-1-000-1915278/)
* April 29: United Kingdom | Death counts updated to reflect deaths outside of hospitals | [Source](https://metro.co.uk/2020/04/29/uk-death-toll-rises-26097-care-homes-included-12628454/) | **Back Distributed**
* May 6: Belgium | 339 new deaths, 229 of which had occured over recent weeks | [Source](http://www.xinhuanet.com/english/2020-05/06/c_139035611.htm)
* June 5: Michigan, US | Release of probable cases and deaths | See [Issue #2704](https://github.com/CSSEGISandData/COVID-19/issues/2704) | **Back Distributed**
* June 8: Chile | Spike of 653 deaths | Historical deaths | [Source](https://www.emol.com/noticias/Nacional/2020/06/07/988430/minsal-muertos-covid19.html)
* June 11: Michigan, US | Michigan started to report probable cases and probable deaths on June 5. ([Source](https://www.michigan.gov/coronavirus/0,9753,7-406-98158-531156--,00.html)) We combined the probable cases into the confirmed cases, and the probable deaths into the deaths. As a consequence, a spike with 5.5k+ cases is shown in our daily cases bar chart.
* June 12: Massachusetts, US | Probable cases released | **Back Distributed**
* June 16: Spain | Revision of historical death count resulting in spike of 1179 cases| [source 1](https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_140_COVID-19.pdf) & [source 2](https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_141_COVID-19.pdf)
* June 16: India | 1672 backlogged deaths from Delhi and Maharashtra | [Source](https://www.hindustantimes.com/india-news/india-s-death-toll-soars-past-10k-backlog-deaths-raise-count-by-437-in-delhi-1-409-in-maharashtra/story-9GNbe7iMBKLsiHtByjRKCJ.html).
* June 17: Chile | Release of 31k unreported cases | See [Issue #2722](https://github.com/CSSEGISandData/COVID-19/issues/2722) | **Back distributed**
* June 23: Delaware, US | Release of some probable deaths and historical cases | See [Issue #2789](https://github.com/CSSEGISandData/COVID-19/issues/2789)
* June 25: New Jersey, US | Release of probable deaths | See [Issue #2763](https://github.com/CSSEGISandData/COVID-19/issues/2763) | **Back distributed**
* July 1: New York City, New York, US | Increase of 682 deaths | [Source](https://www1.nyc.gov/site/doh/covid/covid-19-data.page)
* July 7: Illinois | Incorporation of probable cases and deaths that are being released by the Illinois Department of Health once per week, starting July 3rd. We anticipate weekly spikes in both of these numbers.
* July 12: Phillippines | 141 historical deaths reported | [Source](https://rappler.com/nation/coronavirus-cases-philippines-july-12-2020)
* July 18: Kyrgyzstan | Alteration of probable deaths to include those diagnosed with pneumonia that have not been tested for COVID-19 | [Source](https://www.hrw.org/news/2020/07/21/kyrgyzstan/kazakhstan-new-rules-tallying-covid-19-data)
* July 22: Peru | Addition of 3688 deaths from analyzing historical death records. It is unclear if these are probable deaths or retroactively diagnosed. | [Source](https://www.gob.pe/institucion/minsa/noticias/214828-minsa-casos-confirmados-por-coronavirus-covid-19-ascienden-a-366-550-en-el-peru-comunicado-n-180)
* July 27: Texas, US | Department of State Health Services changed their reporting methodology for COVID-19 deaths, resulting in a roughly 13% increase in reported fatalities from the 26th to the 27th | Details can be found in the press release from the state [here](https://www.dshs.texas.gov/news/releases/2020/20200727.aspx). | **Back distributed for Harris County**
* July 29: Connecticut | Inclusion of 384 historical cases from lab tests "performed during April-June (which) were newly reported to DPH in connection with a transition to electronic reporting by an out of state regional laboratory and for surveillance purposes have been added to the total case and test counts" | [Source](https://portal.ct.gov/-/media/Coronavirus/CTDPHCOVID19summary7292020.pdf)). The 463 spike is consistent with the ct.gov data ([source](https://data.ct.gov/Health-and-Human-Services/COVID-19-Tests-Cases-Hospitalizations-and-Deaths-S/rf3k-f8fg/data).
* July 29: Kazakhstan | Alteration of probable deaths to include those diagnosed with pneumonia that have not been tested for COVID-19 | [Source](https://www.hrw.org/news/2020/07/21/kyrgyzstan/kazakhstan-new-rules-tallying-covid-19-data)
* August 8: Virginia | Spikes in cases are associated with the release of a backlog of testing | [Source](https://wtop.com/virginia/2020/08/recent-surge-in-virginia-covid-19-numbers-due-to-data-backlog)
* August 11: California | Cases are likely to be erratic for the next several days/weeks as a systematic issue with underreporting is being addressed. See the disclaimer posted [here](https://covid19.ca.gov/data-and-tools/): "Note: Due to issues with the state’s electronic laboratory reporting system, these data represent an underreporting of actual positive cases in one single day."
* August 12: Massachusetts | Department of Public Health changed their reporting methodology. The state is no longer reporting county level total cases and deaths daily. Massachusetts is now reporting state level confirmed cases and deaths daily, and are updating state level probable cases and county level confirmed cases weekly. In light of this change by the state, new cases and deaths are being aggregated in "Unassigned, Massachusetts".
* August 14: Peru | Release of 3,658 historical deaths | [Source](https://www.gob.pe/institucion/minsa/noticias/292693-ministerio-de-salud-presento-nueva-actualizacion-de-cifra-de-fallecidos-por-covid-19).
* August 18: Israel | 53 newly idenified nursing home fatalities that occured within July and August | [Source](https://t.me/s/MOHreport/5697)
* August 20: Massachusetts | As previously noted, the Massachusetts Department of Public Health changed their reporting methodology on August 12th (see #3026), dropping their  reporting of daily cumulative confirmed + probable cases and deaths at the county level.  Beginning on August 19th, the state resumed reporting of daily county level data, however the new structure contains confirmed cases, and confirmed + probable deaths.  To accommodate, beginning on August 20th the data we are reporting at the county level will line up with Massachusetts' new reporting (i.e. confirmed cases, confirmed and probable deaths).  Statewide probable cases will be aggregated in the entry for "Unassigned, Massachusetts".  This will unfortunately introduce a drop in total cases at the county level on August 20th as the county level probable cases are shifted to a statewide aggregate.  If and when historical data becomes available we will revise the prior reporting in line with this new definition.
* August 26: Belgium : 352 new deaths and 473 deaths that had either been double counted or misattributed to COVID-19 | [Source 1](https://covid-19.sciensano.be/sites/default/files/Covid19/MORTALITE%20COVID-19%20%E2%80%93%20MISE%20%C3%80%20JOUR%20DES%20DONNEES%20%E2%80%93%2026%20AO%C3%9BT%202020.pdf), [Source 2](https://www.lecho.be/dossiers/coronavirus/le-covid-19-a-fait-moins-de-morts-qu-annonce-en-belgique/10247337.html)
* September 3: Massachusetts | The Massachusetts state government has altered their definition of probable cases ([see source](https://www.mass.gov/doc/covid-19-dashboard-september-2-2020/download)). Prior to September 2, the criteria for probable cases was: if they have a positive antigen test AND have symptoms OR were exposed to someone with COVID; if they have a positive antibody test AND have symptoms OR were exposed to someone with COVID; if they have COVID symptoms AND were exposed to someone with COVID; or if they died and their death certificate lists COVID as a cause of death. Starting September 2, the criteria for probable cases became: if they have a positive antigen test; if they have COVID symptoms AND were exposed to someone with COVID; or if they died and their death certificate lists COVID as a cause of death. The change in definition (and no longer including antibody tests) has resulted in the loss of 8051 cases located in the Unassigned, Massachusetts entry.  **Back Distributed**
* September 4: Illinois | Spike of 5,368 cases | Test backlog | [Source](https://www.nbcchicago.com/news/coronavirus/illinois-reports-5368-new-coronavirus-cases-after-test-backlog-29-additional-deaths/2334290/).
* September 6: Ecuador | Drop of 7953 cases | Definition of confirmed cases from PCR+rapid tests to only PCR tests | [Source](https://www.salud.gob.ec/msp-presenta-actualizacion-de-indicadores-en-infografia-nacional-covid-19/).
* September 7: Ecuador | Inclusion of 3758 probable deaths | The first date where the probable cases are delineated is [September 8](https://twitter.com/Salud_Ec/status/1303463345056616450/photo/1)
* September 15: Alabama | Distributino of probable cases to the county level (previously aggregated at the state level, in unassigned, AL). This resulted in significant increase in cases in nearly all counties. We are working to get the historical distribution of these probable cases from the State, and will update the timeseries accordingly when we do.
* September 15: Arkansas | Addition of 139 probable deaths | [News source](https://katv.com/news/local/arkansas-gov-asa-hutchinson-to-give-covid-19-briefing-09-15-2020).
* September 15-21: Virginia | Progressive inclusion of backlogged deaths into the state total throughout the week | [News source](https://www.wtkr.com/news/coronavirus/local-area-reports-zero-covid-19-deaths-from-sunday-virginia-continues-to-see-steady-decrease-in-test-percent-positivity)
* September 21: Texas, US | Release of large swath of historical cases affecting 25 counties. Please see pinned issue #3143 for full statement from Texas dashboard.
* September 25: North Carolina | Spike of 6000 cases due to inclusion of positive antigen tests as probable cases | **Back Distributed**
* September 29: Alabama | Releases statement that: "Due to technical issues, laboratory errors, and backlogs from onboarding new laboratories, the following dates: 06/02, 06/28, 08/11, 08/25, 09/24, and 09/29 saw an inflation in total number of daily cases" | [Source](https://alpublichealth.maps.arcgis.com/apps/opsdashboard/index.html#/6d2771faa9da4a2786a509d82c8cf0f7) - See Tab 6
* October 1: Argentia | Release of 3050 previously unidentified deaths in the Buenos Aires province | [Official report](https://www.argentina.gob.ar/sites/default/files/01-10-20-reporte-vespertino-covid-19.pdf)
* October 2: New Hampshire | 139 probable cases (positive via antigen test) dating back to July | [Official press release](https://www.nh.gov/covid19/news/documents/covid-19-update-10022020.pdf)
* October 3-4: United Kingdom | Notice on United Kingdom's dashboard states "Due to a technical issue, which has now been resolved, there has been a delay in publishing a number of COVID-19 cases to the dashboard in England. This means the total reported over the coming days will include some additional cases from the period between 24 September and 1 October, increasing the number of cases reported." We anticipate this will lead to significantly higher case numbers being reported. ([Data source](https://coronavirus.data.gov.uk/)). On October 4, the United Kingdom released the following statement on their dashboard: "The cases by publish date for 3 and 4 October include 15,841 additional cases with specimen dates between 25 September and 2 October — they are therefore artificially high for England and the UK."
* October 5: Mexico | Alteration of case definition to include those epidemiologically linked and symptomatic but lacking test confirmation. The result of this change is an increase of 24,698 historical cases and 2609 historical deaths, both going back to the beginning of the pandemic | [News source with official press conference embedded](https://www.eluniversal.com.mx/nacion/coronavirus-5-de-octubre-mexico-suma-789-mil-casos-de-covid-y-81-mil-muertes)
* October 7: Fayette County, Kentucky | Release of 1472 historical cases stretching over the previous month and a half. These are a portion of 1900 backlogged cases for the county, and it is likely this will be included in the proceeding days | [County source](https://www.facebook.com/LFCHD/posts/10159412744354739), [News source with Governor's press conference](https://www.wkyt.com/2020/10/07/watch-live-gov-beshear-gives-update-on-covid-19/)
* October 12: Missouri } spike in cases is due to a database error. We are monitoring the dashboard and will redistribute if the error is fixed | [News report](https://www.stltoday.com/lifestyles/health-med-fit/health/missouri-health-department-says-5-000-case-increase-was-reporting-error/article_0021cc3b-21a4-5c6f-8887-48aa6e2087bd.html)
* October 18: Navajo County, Arizona | Reduction of -52 cases | Reconciliation of database entry errors | [Source](https://twitter.com/NavajoCountyAZ/status/1317506622281850881)
* October 23: Alabama, US | Inclusion of 2565 backlogged antigen positives in Mobile and 1182 backlogged antigen and PCR tests from around the state | [Source](https://alpublichealth.maps.arcgis.com/apps/opsdashboard/index.html#/6d2771faa9da4a2786a509d82c8cf0f7) "The Alabama Department of Public Health processed a backlog of 2565 positive antigen results from a facility in Mobile on October 22. These will be classified as “probable” COVID-19 cases reported on 10/22/20 even though the tests were performed during June through October 18, 2020. The Alabama Department of Public Health processed a backlog of 1182 positive results from a variety of facilities all over Alabama. A majority of these will be classified as “probable” COVID-19 cases reported on 10/23/20 even though the tests were performed during April through September."
* October 23: Los Angeles County, California, US | Estimated 2000 backlogged cases included in daily report | [Source](https://github.com/CSSEGISandData/COVID-19/issues/3267)
* November 4: Spain | Inclusion of 5,105 cases and 1,326 deaths that occurred prior to May 11 | [Source](https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/Actualizacion_243_COVID-19.pdf)
* November 3: Georgia | Inclusion of 29,937 antigen tests distributed over unknown period of days | We have contacted the state health department to obtain a back distribution [Media source](https://www.wrbl.com/news/georgia-news/update-georgia-reports-364589-confirmed-covid-19-cases-statewide-with-6440-in-columbus/) 


## Large-scale back distributions
This section will serve to notify developers when we are able to successfully backdistribute any of the large instances of retrospective reporting.
Generalized format:
Date: Location | File | Change | Data source for change
* April 24: New York City, New York, US (April 23) | Distribution of probable deaths from March 12 to April 24 (See errata.csv line 104) | 
* April 29: United Kingdom (April 29) | time_series_covid19_deaths_global.csv | Distribution of deaths outside of hospital | Official government website
* June 12: Massachusetts, US (June 12) | Probable cases back distributed | Source is [here](https://www.mass.gov/info-details/covid-19-response-reporting).
* June 13: Michigan | Through data provided by the Michigan Department of Health and Human Service’s (MDHHS) Communicable Disease Division, we were able to appropriately distribute the probable cases MDHHS began reporting on June 5th.
* July 1: New York City, New York (July 1) | time_series_covid19_deaths_us.csv | Probable deaths back distributed via tables on [nychealth GitHub](https://github.com/nychealth/coronavirus-data/blob/master/deaths/probable-confirmed-dod.csv).
* July 26: Chile (June 17) | Back distribution of probable and previously non-notified cases (all prior to June 17) | time_series_covid19_confirmed_global.csv | Data from [this repository] (https://github.com/MinCiencia/Datos-COVID19) managed by the Ministry of Science was used for the correction. Specifically, data from [product 45] CasosConfirmadosPorComunaHistorico_std.csv and CasosNoNotificadosPorComunaHistorico_std.csv was accessed on July 26 and the most current version of the documents at that time were used for the correction. For CasosConfirmadosPorComunaHistorico_std.csv, this was July 22nd. Cases were added to the day at the end of their respective epidemiological week. 
* August 2: New Jersey, US (June 25) | time_series_covid19_deaths_US.csv | Redistribution of probable deaths into Unassigned, New Jersey
* August 6: Harris County, Texas (Texas July 27) | In line with alteration to reporting standards, time series for coronavirus deaths in Harris County has been updated from 3/7 to 8/5/2020 | Details can be found in the press release from the state [here](https://www.dshs.texas.gov/news/releases/2020/20200727.aspx).
* September 22: Massachusetts (September 3) | Reconciliation of changes to probable cases | Detailed [here](https://github.com/CSSEGISandData/COVID-19/issues/3146)
* October 1: North Carolina | Back distribution of probable cases | See [Issue #3183](https://github.com/CSSEGISandData/COVID-19/issues/3183#ref-commit-663bcf9)

## Irregular Update Schedules
As the pandemic has progressed, several locations have altered their reporting schedules to no longer provide daily updates. As these locations are identified, we will list them in this section of the README. We anticipate that these irregular updates will cause cyclical spikes in the data and smoothing algorithms should be applied if the data is to be used for modeling.

United States
* Rhode Island: Not updating case, death, or recovered data on the weekends. Releasing county level cases and deaths once per week.
* Conneticut: Not updating case, death, or recovered data on the weekends.
* Illinois: Releasing probable cases once per week.
* District of Columbia: Not updating on the weekends. Periodically updated using data available [here](https://coronavirus.dc.gov/data).
* Louisiana: Not updating on the weekends.
* Michigan: No case data provided for August 21. 
* Kansas: No data for the weekend of August 22-23.
* Guam: Not reporting data on weekends.
* Michigan: Not providing death data on Sundays.
* Florida: Did not update on weekend for October 10-11.
* Washington: Did not update October 10-12 due to data entry issue. Back distribution is not available.

International
* Sweden: Not updating case, death, or recovered data Saturday-Monday. Updates expected Tuesdays and Fridays.
* Spain: Not updating case or death data on the weekends (and is not currently providing recoveries at any time)
* Nicaragua: Releasing case, death, and recovered data once per week.
* UK: daily death toll paused on July 18. ([GOV.UK](https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public#number-of-cases) and [Reuters](https://www.reuters.com/article/us-health-coronavirus-britain-casualties-idUSKCN24J0GC))
* France: No longer releasing case, hospitalization, or death data on the weekends. Please see [Tableau dashboard](https://dashboard.covid19.data.gouv.fr/vue-d-ensemble?location=FRA). 
* Denmark: Not updating case, death, or recovered data on the weekends.
* France: No update to deaths or recoveries for the weekend of August 8 and 9.
* UK (2): Technical difficulties with the national dashboard are resulting in no update for August 11. [Source](https://twitter.com/phe_uk/status/1293245784599781376?s=21). Corrected on August 12.
* Luxembourg: Not providing actionable data on weekends.
* Mexico: Beginning November 10, recoveries are available at the national level only and will be grouped in the "Unassigned, Mexico" entry.


---
## [UID Lookup Table Logic](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv)

1.	All countries without dependencies (entries with only Admin0).
  *	None cruise ship Admin0: UID = code3. (e.g., Afghanistan, UID = code3 = 4)
  *	Cruise ships in Admin0: Diamond Princess UID = 9999, MS Zaandam UID = 8888.
2.	All countries with only state-level dependencies (entries with Admin0 and Admin1).
  *	Demark, France, Netherlands: mother countries and their dependencies have different code3, therefore UID = code 3. (e.g., Faroe Islands, Denmark, UID = code3 = 234; Denmark UID = 208)
  *	United Kingdom: the mother country and dependencies have different code3s, therefore UID = code 3. One exception: Channel Islands is using the same code3 as the mother country (826), and its artificial UID = 8261.
  *	Australia: alphabetically ordered all states, and their UIDs are from 3601 to 3608. Australia itself is 36.
  *	Canada: alphabetically ordered all provinces (including cruise ships and recovered entry), and their UIDs are from 12401 to 12415. Canada itself is 124.
  *	China: alphabetically ordered all provinces, and their UIDs are from 15601 to 15631. China itself is 156. Hong Kong, Macau and Taiwan have their own code3.
  *	Germany: alphabetically ordered all admin1 regions (including Unknown), and their UIDs are from 27601 to 27617. Germany itself is 276.
  * Italy: UIDs are combined country code (380) with `codice_regione`, which is from [Dati COVID-19 Italia](https://github.com/pcm-dpc/COVID-19). Exceptions: P.A. Bolzano is 38041 and P.A. Trento is 38042.
3.	The US (most entries with Admin0, Admin1 and Admin2).
  *	US by itself is 840 (UID = code3).
  *	US dependencies, American Samoa, Guam, Northern Mariana Islands, Virgin Islands and Puerto Rico, UID = code3. Their Admin0 FIPS codes are different from code3.
  *	US states: UID = 840 (country code3) + 000XX (state FIPS code). Ranging from 8400001 to 84000056.
  *	Out of [State], US: UID = 840 (country code3) + 800XX (state FIPS code). Ranging from 8408001 to 84080056.
  *	Unassigned, US: UID = 840 (country code3) + 900XX (state FIPS code). Ranging from 8409001 to 84090056.
  *	US counties: UID = 840 (country code3) + XXXXX (5-digit FIPS code).
  *	Exception type 1, such as recovered and Kansas City, ranging from 8407001 to 8407999.
  *	Exception type 2, Bristol Bay plus Lake Peninsula replaces Bristol Bay and its FIPS code. Population is 836 (Bristol Bay) + 1,592 (Lake and Peninsula) = 2,428 (Bristol Bay plus Lake Peninsula). 2148 (Hoonah-Angoon) + 579 (Yakutat) = 2727 (Yakutat plus Hoonah-Angoon). UID is 84002282, the same as Yakutat. ~~New York City replaces New York County and its FIPS code. New York City popluation is calculated as Bronx (1,418,207) + Kings (2,559,903) + New York (1,628,706) + Queens (2,253,858) + Richmond (476,143) = NYC (8,336,817). (updated on Aug 31)~~ 
  *	Exception type 3, Diamond Princess, US: 84088888; Grand Princess, US: 84099999.
  * Exception type 4, municipalities in Puerto Rico are regarded as counties with FIPS codes. The FIPS code for the unassigned category is defined as 72999.
4. Population data sources.
 * United Nations, Department of Economic and Social Affairs, Population Division (2019). World Population Prospects 2019, Online Edition. Rev. 1. https://population.un.org/wpp/Download/Standard/Population/
 * eurostat: https://ec.europa.eu/eurostat/web/products-datasets/product?code=tgs00096
 * The U.S. Census Bureau: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
 * Mexico population 2020 projection: [Proyecciones de población](http://sniiv.conavi.gob.mx/(X(1)S(kqitzysod5qf1g00jwueeklj))/demanda/poblacion_proyecciones.aspx?AspxAutoDetectCookieSupport=1)
* Brazil 2019 projection: ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/
* Peru 2020 projection: https://www.citypopulation.de/en/peru/cities/
* India 2019 population: http://statisticstimes.com/demographics/population-of-indian-states.php
* The Admin0 level population could be different from the sum of Admin1 level population since they may be from different sources.

Disclaimer: \*The names of locations included on the Website correspond with the official designations used by the U.S. Department of State. The presentation of material therein does not imply the expression of any opinion whatsoever on the part of JHU concerning the legal status of any country, area or territory or of its authorities. The depiction and use of boundaries, geographic names and related data shown on maps and included in lists, tables, documents, and databases on this website are not warranted to be error free nor do they necessarily imply official endorsement or acceptance by JHU.
