library(deSolve)
library(FME)
library(readxl)
library(openxlsx)
library(writexl)
library(dplyr)
library(stringr)
ZF.model <- function(t,initial_v, parameters) {
  with(as.list(c(initial_v, parameters)),{
    Km = NULL
    V_art = sc_blood * BW * art_ven_frac
    V_ven = sc_blood * BW - V_art
    V_liv = sc_liv * BW
    V_gon = sc_gon * BW
    V_fat = sc_fat * BW
    V_git = sc_git * BW
    V_brain = sc_brain * BW
    V_kidney = sc_kidney * BW
    V_skin = sc_skin * BW
    V_rp = sc_rp * BW
    V_pp = BW-(V_art+V_ven+V_liv+V_gon+V_fat+V_git+V_brain+V_kidney+V_skin+V_rp)
    V_total = V_art+V_ven+V_liv+V_gon+V_fat+V_git+V_brain+V_kidney+V_skin+V_rp+V_pp
    V_bal = BW- V_total
    Fliv = liv_frac * Fcard
    Fgon = gon_frac * Fcard
    Fgit = git_frac * Fcard
    Ffat = fat_frac * Fcard
    Fbrain = brain_frac * Fcard
    Fkidney = kidney_frac * Fcard
    Fskin = skin_frac * Fcard
    Frp = rp_frac * Fcard
    Fpp = Fcard-(Fliv+Fgon+Fgit+Ffat+Fbrain+Fkidney+Fskin+Frp)
    Ftotal = Fliv+Fgon+Fgit+Ffat+Fbrain+Fkidney+Fskin+Frp+Fpp
    Fbal = Fcard- Ftotal
    Co2w = ((-0.24 * TC_c + 14.04) * Sat)/100
    Fwater = VO2/(OEE*Co2w) 
    Cl_plasma_sc = Cl_plasma*BW
    Cl_liv=Cl
    if ( !is.na(Vmax)) { Vmax_sc = Vmax * V_liv } 
    if ( !is.na(Cl_liv)) { Cl_sc = Cl_liv * V_liv }
    if (pKa_a_pred != 0 && pKa_b_pred == 0) {
      aa <- 1
    } else if (pKa_a_pred == 0 && pKa_b_pred != 0) {
      aa <- -1
    } else if (pKa_a_pred != 0 && pKa_b_pred != 0) {
      aa <- 1
    } else {
      stop("Conditions do not meet expectations")
    }
    fngill=1/(1+(10^(aa*(pH-pka)))) #acid i=1,base i=-1
    fnfish=1/(1+(10^(aa*(7.4-pka))))
    log_Kowion= log_Kow - 3.5
    Dowgill=fngill*(10^log_Kow)+(1-fngill)*(10^log_Kowion)
    Dow= fnfish*(10^log_Kow)+(1-fnfish)*(10^log_Kowion)
    rwater = VO2/(0.71*Co2w)*(1/((1000^0.25)*(BW^0.75)))
    rblood =  Fcard* PBW* (1/((1000^0.25)*(BW^0.75)))
    kx = ((BW/1000)^0.75/(0.0028+(68/Dowgill)+(1/rwater)+(1/rblood)))*1000
    dA_admin_gil = kx * ((A_water)/V_water) 
    dA_lumen_GIT = (A_bile* K_BG-Ku*A_lumen_GIT-Ke_feces*A_lumen_GIT)
    dA_admin_git = 0
    dA_excr_gil = kx * (((A_ven/V_ven)*Unbound_fraction)/PBW) 
    dV_urine = urine_rate* BW
    dA_urine = Ke_urine * A_kidney
    dA_urine_cum = Ke_urine * A_kidney
    dA_feces = A_lumen_GIT * Ke_feces
    dA_excr_water = dA_excr_gil + dA_urine_cum + dA_feces
    dA_bile = (Ke_bile * A_liv)-(A_bile * K_BG)
    dA_met_liv = if ( is.na(Cl_liv))
    {(Vmax_sc*((A_liv/V_liv)/Plivb))/(Km+(A_liv/V_liv)/Plivb)} 
    else{(Cl_sc*((A_liv/V_liv)/Plivb))}
    dA_met_plasma = Cl_plasma_sc* A_ven
    dA_Elimination = dA_met_liv+ dA_met_plasma
    dA_art = (Fcard*(A_ven / V_ven)-Fliv*(A_art/V_art)-Ffat*(A_art/V_art)-Fskin*(A_art/V_art)
              -Fgon*(A_art /V_art)-Fgit*(A_art/V_art)- Fbrain*(A_art/V_art)-Fkidney*(A_art/V_art)
              -Frp*(A_art /V_art)-Fpp*(A_art /V_art))
    dA_ven = (dA_admin_gil-dA_excr_gil-Fcard*(A_ven/V_ven) 
              +Ffat*((A_fat/V_fat)/Pfb)+a_Fs*Fskin*((A_skin/V_skin)/Pskinb)
              +Fbrain*((A_brain /V_brain)/Pbb)+(Fliv+Fgon+Fgit+Frp)*((A_liv/V_liv)/Plivb)+
                +(Fkidney+(1-a_Fs)*Fskin+(1-a_Fpp)*Fpp)*((A_kidney/V_kidney)/Pkidb)+a_Fpp*Fpp*((A_pp/V_pp)/Pppb))
    dA_fat = Ffat*(A_art/V_art)- Ffat* ((A_fat/V_fat)/Pfb)
    dA_skin = Fskin*(A_art/V_art) - Fskin*((A_skin/V_skin)/Pskinb)
    dA_git = Ku*A_lumen_GIT + Fgit*(A_art/V_art) - Fgit*((A_git/V_git)/Pgitb) 
    dA_gon = Fgon*(A_art/V_art) - Fgon*((A_gon/V_gon)/Pgonb)
    dA_brain = Fbrain*(A_art/V_art) - Fbrain*((A_brain/V_brain)/Pbb)
    dA_liv = (Fliv*(A_art/V_art)+ Fgon*((A_gon/V_gon)/Pgonb)+ Fgit*((A_git/V_git)/Pgitb)+Frp*((A_rp/V_rp)/Prpb)
              -(Fliv+Fgon+Fgit+Frp)*((A_liv/V_liv)/Plivb)
              -Ke_bile*A_liv-dA_met_liv) 
    
    dA_kidney = (Fkidney*(A_art/V_art)+(1-a_Fs)*Fskin*((A_skin/V_skin)/Pskinb)+(1- a_Fpp)*Fpp*((A_pp/V_pp)/Pppb)
                 -(Fkidney+(1-a_Fs)*Fskin+(1-a_Fpp)*Fpp)*((A_kidney/V_kidney)/Pkidb) -dA_urine)
    
    dA_rp = Frp*(A_art/V_art)- Frp* ((A_rp/V_rp)/Prpb)
    dA_pp = Fpp*(A_art/V_art)- Fpp* ((A_pp/V_pp)/Pppb)
    dA_water = (dA_excr_water-dA_admin_gil)
    V_tot= V_ven+V_art+V_liv+V_fat+V_gon+V_git+V_brain+V_kidney+V_rp+V_pp+V_skin
    A_body_tot = (A_art+A_ven+A_bile+A_liv+A_fat+A_gon+A_git+A_brain+A_kidney+A_skin+A_rp+A_pp)
    A_input = A_admin_gil+A_admin_git+A_iv
    A_elim = A_met_liv+A_met_plasma+A_excr_water+A_lumen_GIT
    A_tot_sys = A_water+A_input+A_body_tot
    Mass_bal_sys = A_input-A_body_tot-A_elim
    
    C_art = A_art/V_art
    C_ven = A_ven/V_ven
    C_liv = (A_liv+A_bile)/V_liv
    C_fat = A_fat/V_fat
    C_gon = A_gon/V_gon
    C_git = A_git/V_git
    C_brain = A_brain/V_brain
    C_kidney = A_kidney/ V_kidney 
    C_skin = A_skin/V_skin
    C_rp = A_rp/V_rp
    C_pp = A_pp/V_pp
    C_carcass = (A_body_tot-A_gon-A_liv-A_brain)/(V_tot-V_gon-V_liv-V_brain)
    C_plasma = C_art / Ratio_blood_plasma
    AA_whole_body = ((A_art + A_ven + A_liv + A_gon + A_brain + A_fat + A_skin + A_git + A_kidney + A_pp + A_rp)
                     / (V_art + V_ven + V_liv + V_gon + V_brain + V_fat + V_skin + V_git + V_kidney + V_pp + V_rp))#换公式了
    AA_liv = (A_liv+A_bile)/BW
    AA_fat = A_fat/BW
    AA_gon = A_gon/BW
    AA_git = A_git/BW
    AA_brain = A_brain/BW
    AA_kidney = A_kidney/BW 
    AA_skin = A_skin/BW
    AA_rp = A_rp/BW
    AA_pp = A_pp/BW
    AA_carcass = (A_body_tot-A_gon-A_liv-A_brain)/BW
    
    list(c(dA_water, dA_excr_water, 
           dA_admin_gil,dA_admin_git,dA_excr_gil,dV_urine,dA_urine, dA_urine_cum, 
           dA_feces,dA_lumen_GIT, dA_met_liv, dA_met_plasma, dA_art, dA_ven, dA_liv, 
           dA_kidney, dA_fat, dA_skin, dA_gon, dA_git, dA_brain, dA_rp,dA_pp,
           dA_bile, dA_Elimination),
         "C_art"= C_art ,
         "C_ven" = C_ven,
         "C_liv" = C_liv,
         "C_fat" = C_fat,
         "C_gon" = C_gon,
         "C_git" = C_git,
         "C_brain" = C_brain,
         "C_kidney" = C_kidney,
         "C_skin" = C_skin,
         "C_rp" = C_rp,
         "C_pp" = C_pp,
         "C_carcass" = C_carcass,
         "C_plasma" = C_plasma,
         "AA_whole_body"=AA_whole_body
    )
  })}

####PARTITION COEFFICIENTS 
PC_qsar_model = function(
    Dow,Funbound
){
  PC_QSAR_blood_water = 0.008 *0.3* Dow + 0.007*2.0*(Dow^0.94) + 0.134*2.9*(Dow^0.63) + 0.851
  PC_blood_water = PC_QSAR_blood_water
  PC_QSAR_liver = (0.037*0.3*Dow + 0.025*2.0*(Dow^0.94)+ 0.234*2.9 *(Dow^0.63) +0.704)/PC_blood_water
  PC_QSAR_gonads = (0.042*0.3*Dow + 0.024*2.0*(Dow^0.94)+ 0.262 *2.9*(Dow^0.63) +0.672)/PC_blood_water
  PC_QSAR_brain = (0.047*0.3*Dow + 0.043*2.0*(Dow^0.94)+ 0.228 *2.9*(Dow^0.63) +0.682)/PC_blood_water
  PC_QSAR_fat = (0.884*0.3*Dow + 0.009*2.0*(Dow^0.94)+ 0.048 *2.9*(Dow^0.63) +0.059)/PC_blood_water 
  PC_QSAR_skin = (0.037*0.3*Dow + 0.012*2.0*(Dow^0.94)+ 0.259 *2.9*(Dow^0.63) +0.692)/PC_blood_water
  PC_QSAR_GIT = (0.052*0.3*Dow + 0.023*2.0*(Dow^0.94)+ 0.255 *2.9*(Dow^0.63) +0.67)/PC_blood_water
  PC_QSAR_kidney = (0.059*0.3*Dow + 0.023*2.0*(Dow^0.94)+ 0.215 *2.9*(Dow^0.63) +0.703)/PC_blood_water
  PC_QSAR_rp = (0.066*0.3*Dow + 0.008*2.0*(Dow^0.94)+ 0.338 *2.9*(Dow^0.63) +0.587)/PC_blood_water
  PC_QSAR_pp = (0.024*0.3*Dow + 0.009*2.0*(Dow^0.94)+ 0.199*2.9 *(Dow^0.63) +0.769)/PC_blood_water
  Plivb = PC_QSAR_liver
  Pgonb = PC_QSAR_gonads 
  Pbb = PC_QSAR_brain
  Pfb = PC_QSAR_fat
  Pskinb = PC_QSAR_skin
  Pgitb = PC_QSAR_GIT 
  Pkidb = PC_QSAR_kidney
  Prpb = PC_QSAR_rp
  Pppb = PC_QSAR_pp
  PBW = PC_blood_water*Funbound

  return( c( "PBW" = PBW, "Pgitb" = Pgitb,"Pgonb" = Pgonb,"Plivb" = Plivb,
             "Pfb" = Pfb, "Pbb" = Pbb, "Pkidb"=Pkidb,"Pskinb"= Pskinb, "Pppb"= Pppb, "Prpb"= Prpb ))
}


########OUTPUT FUNCTION
output <- function(parms){
  method = as.character("lsoda") 
  start = parms[["start"]]
  stop = parms[["stop"]]
  Res_times = c(1
                
  )
  times = sort( unique( c(seq(start,stop,0.1), Res_times)))
  period = NA # days between two doses for oral
  frac_renewed = 0 # fraction of the water of the aquaria renewed
  time_final_dose = parms[["time_final_dose"]]
  time_first_dose = parms[["time_first_dose"]]
  TC_c = parms[["TC_c"]] 
  Sat = 0.9 
  OEE = parms[["OEE"]]
  BW = parms[["BW"]]
  A_iv = parms[["A_iv"]]
  food = parms[["food"]]
  frac_absorbed = parms[["frac_absorbed"]]
  Dose_water = parms[["Dose_water"]] 
  V_water = parms[["V_water"]]
  art_ven_frac = parms[["art_ven_frac"]]
  sc_blood = parms[["sc_blood"]]
  sc_liv = parms[["sc_liv"]]
  sc_gon = parms[["sc_gon"]]
  sc_fat = parms[["sc_fat"]]
  sc_git = parms[["sc_git"]]
  sc_brain = parms[["sc_brain"]]
  sc_kidney = parms[["sc_kidney"]]
  sc_skin = parms[["sc_skin"]]
  sc_rp = parms[["sc_rp"]]
  sc_pp = (1-sc_blood-sc_liv-sc_gon-sc_fat-sc_git-sc_brain-sc_kidney-sc_skin-sc_rp)
  liv_frac = parms[["liv_frac"]]
  gon_frac = parms[["gon_frac"]]
  git_frac = parms[["git_frac"]]
  fat_frac = parms[["fat_frac"]]
  brain_frac = parms[["brain_frac"]]
  kidney_frac = parms[["kidney_frac"]]
  skin_frac = parms[["skin_frac"]]
  rp_frac = parms[["rp_frac"]]
  pp_frac = (1- liv_frac- gon_frac- git_frac- fat_frac- brain_frac- kidney_frac- skin_frac-rp_frac )
  Dow=parms[["Dow"]] 
  pH = parms[["pH"]] 
  pka= parms[["pka"]] 
  log_Kow = parms[["log_Kow"]] 
  Unbound_fraction = parms[["Unbound_fraction"]]
  Ratio_blood_plasma = parms[["Ratio_blood_plasma"]]
  Ku = parms[["Ku"]] #Oral absorption 
  Ke_urine = parms[["Ke_urine"]] 
  Ke_feces = parms[["Ke_feces"]] 
  urine_rate = parms[["urine_rate"]]
  urination_interval = NA
  
  
  PC_BPA = PC_qsar_model(Dow = Dow, Funbound = Unbound_fraction)
 
  parameters = c(
    Dose_water = Dose_water,
    A_iv = A_iv,
    food = food, 
    V_water = V_water, 
    BW = BW,
    TC_c =TC_c,
    Sat = Sat,
    OEE = parms[["OEE"]],
    a_Fs = parms[["a_Fs"]],
    a_Fpp = parms[["a_Fpp"]],
    Fcard =parms[["Fcard"]],
    VO2 =parms[["VO2"]],
    aa =parms[["aa"]],
    pKa_a_pred=parms[["pKa_a_pred"]],
    pKa_b_pred=parms[["pKa_b_pred"]],
    Ku = parms[["Ku"]],
    Ke_urine = parms[["Ke_urine"]], 
    Ke_feces = parms[["Ke_feces"]], 
    Ke_bile = parms[["Ke_bile"]],
    K_BG = parms[["K_BG"]], 
    urine_rate = parms[["urine_rate"]],
    urination_interval = parms[["urination_interval"]],
    Cl = parms[["Cl"]],
    sc_blood = parms[["sc_blood"]],
    sc_liv = parms[["sc_liv"]],
    sc_gon = parms[["sc_gon"]],
    sc_fat = parms[["sc_fat"]],
    sc_git = parms[["sc_git"]],
    sc_brain = parms[["sc_brain"]],
    sc_kidney = parms[["sc_kidney"]],
    sc_skin = parms[["sc_skin"]],
    sc_rp = parms[["sc_rp"]],
    sc_pp = parms[["sc_pp"]],
    liv_frac = parms[["liv_frac"]],
    gon_frac = parms[["gon_frac"]],
    git_frac = parms[["git_frac"]],
    fat_frac = parms[["fat_frac"]],
    brain_frac = parms[["brain_frac"]],
    art_ven_frac = parms[["art_ven_frac"]],
    kidney_frac = parms[["kidney_frac"]],
    skin_frac = parms[["skin_frac"]],
    rp_frac = parms[["rp_frac"]],
    pp_frac = parms[["pp_frac"]],
    Vmax = parms[["Vmax"]],
    Cl_plasma = parms[["Cl_plasma"]],
    frac_absorbed = parms[["frac_absorbed"]],
    log_Kow = parms[["log_Kow"]],
    pH= parms[["pH"]],
    pka= parms[["pka"]],
    Dow= parms[["Dow"]],
    Unbound_fraction = parms[["Unbound_fraction"]],
    Ratio_blood_plasma = parms[["Ratio_blood_plasma"]],
    art_ven_frac = parms[["art_ven_frac"]],

    PBW = PC_BPA[["PBW"]],
    Plivb = PC_BPA[["Plivb"]],
    Pgonb = PC_BPA[["Pgonb"]],
    Pbb = PC_BPA[["Pbb"]],
    Pfb = PC_BPA[["Pfb"]],
    Pskinb = PC_BPA[["Pskinb"]],
    Pgitb = PC_BPA[["Pgitb"]],
    Pkidb = PC_BPA[["Pkidb"]],
    Prpb = PC_BPA[["Prpb"]],
    Pppb = PC_BPA[["Pppb"]]
  )
  
  initial_v = c(
    
    A_water = (Dose_water*V_water),
    A_excr_water = 0,
    A_admin_gil = 0,
    A_admin_git = (food*frac_absorbed),
    A_excr_gil = 0,
    V_urine = 0,
    A_urine = 0,
    A_urine_cum = 0,
    A_feces = 0,
    A_lumen_GIT = (food*frac_absorbed),
    A_met_liv = 0,
    A_met_plasma = 0,
    A_art = 0, 
    A_ven = A_iv,
    A_liv = 0,
    A_kidney = 0,
    A_fat = 0,
    A_skin = 0,
    A_gon = 0,
    A_git = 0,
    A_brain = 0,
    A_rp = 0,
    A_pp = 0,
    A_bile = 0,
    Elimination = 0)

  if (!is.na(urination_interval)){
    events_urine <- list(data = rbind(data.frame(var = c("V_urine"), 
                                                 time = seq(times[1], rev(times)[1] , by=urination_interval), 
                                                 value = 0, 
                                                 method = c("replace")),
                                      data.frame(var = c("A_urine"), 
                                                 time = seq(times[1], rev(times)[1] , by=urination_interval), 
                                                 value = 0, 
                                                 method = c("replace"))))
  }else{ events_urine <- NULL }

  events_repeated_f <- if (!is.na(period) & food!=0 ) {

    if (is.null(time_final_dose)){ time_final_dose <- floor(max(times)/period)*period}
  
    if (!is.null(time_first_dose)) { initial_v["A_admin_git"]<-0 ; initial_v["A_lumen_GIT"]<-0 }

    if (is.null(time_first_dose)){ time_first_dose = period }

    events_repeated_f <- list(data = rbind(data.frame(var = c("A_lumen_GIT"), 
                                                      time = seq(time_first_dose, time_final_dose , by=period), 
                                                      value = 
                                                        as.numeric(c(parameters["frac_absorbed"]*parameters["food"])), 
                                                      method = c("add")),
                                           data.frame(var = c("A_admin_git"), 
                                                      time = seq(time_first_dose, time_final_dose , by=period), 
                                                      value = 
                                                        as.numeric(c(parameters["frac_absorbed"]*parameters["food"])), 
                                                      method = c("add")))) 
  } 
  events_repeated_w <- NULL
  if (!is.na(period) & Dose_water!=0 ) {

    if (is.null(time_final_dose)){ time_final_dose <- floor(max(times)/period)*period}

    if (!is.null(time_first_dose)) { initial_v["A_water"]<-0 }

    if (is.null(time_first_dose)){ time_first_dose = period }

    events_repeated_w <- list(data = rbind(data.frame(var = c("A_water"), 
                                                      time = seq(time_first_dose, time_final_dose , by=period), 
                                                      value = (1-frac_renewed),
                                                      method = c("multiply")),
                                           data.frame(var = c("A_water"), 
                                                      time = 0.0000001+seq(time_first_dose, time_final_dose , by=period), 
                                                      value = as.numeric( parameters["Dose_water"]* frac_renewed), 
                                                      method = c("add")),
                                           data.frame(var = c("A_water"), 
                                                      time = time_final_dose, 
                                                      value = 0, 
                                                      method = c("multiply"))
                                           
    )) 
  }
  if ((is.na(period))& (Dose_water!=0) ){
    events_repeated_w <- list(data = rbind(data.frame(var = c("A_water"), 
                                                      time = time_final_dose, 
                                                      value = 0, 
                                                      method = c("multiply"))))
  }
  
  
  events<-list(data = rbind(events_repeated_f[[1]], events_repeated_w[[1]], events_urine[[1]]))

  solution <- ode(times = times, y = initial_v, func = ZF.model, parms = parameters, method = method, 
                  events = events)
  return(solution)
}


data_path <- "D:/jupyternotebook/python/code/HM-PBTK-master/data.xlsx"
if (!file.exists(data_path)) {
  stop("Please modify the path to the data file as appropriate")
}

base_dir <- dirname(data_path)
output_dir <- file.path(base_dir, "output", "dl")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

parameters_df <- read_excel(data_path, sheet = "dl")
parameters_df <- parameters_df %>%
  rename(BW = `body_weight(g)`, BL = `body_length(cm)`, stop=`total_time(day)`,time_final_dose=`exposure_time(day)`,
         TC_c=`water_temperature`,Dose_water=`dose_water(μg/mL)`,pH=`water_pH` )

required_columns <- c("stop", "time_final_dose", "TC_c", "Dose_water", "BW", "aa", "pKa_a_pred","pKa_b_pred","Fcard", "VO2","Unbound_fraction", "Cl","logKow", "Dow", "pKa","pH",
                      "sc_blood",
                      "sc_liver",
                      "sc_gonads",
                      "sc_fat",
                      "sc_GIT",
                      "sc_brain",
                      "sc_kidney",
                      "sc_skin",
                      "sc_rpt",
                      "sc_ppt",
                      "liver_frac",
                      "gonads_frac",
                      "GIT_frac",
                      "fat_frac",
                      "brain_frac",
                      "kidney_frac",
                      "skin_frac",
                      "rpt_frac",
                      "ppt_frac",
                      "water_liver",
                      "water_brain",
                      "water_gonads",
                      "water_fat",
                      "water_skin",
                      "water_GIT",
                      "water_kidney",
                      "water_rpt",
                      "water_ppt",
                      "lipids_liver",
                      "lipids_brain",
                      "lipids_gonads",
                      "lipids_fat",
                      "lipids_skin",
                      "lipids_GIT",
                      "lipids_kidney",
                      "lipids_rpt",
                      "lipids_ppt" 
                      
)

number_of_rows <- nrow(parameters_df)

for (i in 1:number_of_rows) {

  row_values <- parameters_df[i, required_columns]
  
  if (all(sapply(row_values, is.numeric))) {
   
    parms <- c(
      start = 0,
      time_first_dose = 0,
      A_iv = 0,
      food = 0,
      V_water = 10E12, # Water compartment is set very large to account for flow through and make changes in water c insignificant. Like suggested by Remy Beaudouin #aquarium size in ml --> 100L
      a_Fs = 0.1, 
      a_Fpp = 0.4,
      OEE = 0.71,
      art_ven_frac = 0.33 ,
      Ke_urine = 0,
      urine_rate = 0,
      urination_interval = NA,
      Ku = 0,
      Ke_feces = 0,
      Ke_bile = 0, 
      K_BG = 0, 
      Vmax = NA,
      Cl_plasma = 0,
      frac_absorbed = 1, 
      Ratio_blood_plasma = 1,
      stop = row_values[["stop"]],
      time_final_dose = row_values[["time_final_dose"]],
      TC_c = row_values[["TC_c"]],
      Dose_water = row_values[["Dose_water"]],
      BW = row_values[["BW"]],
      Fcard = row_values[["Fcard"]],
      VO2=row_values[["VO2"]],
      aa=row_values[["aa"]],
      pKa_a_pred=row_values[["pKa_a_pred"]],
      pKa_b_pred=row_values[["pKa_b_pred"]],
      Cl=row_values[["Cl"]],
      Unbound_fraction = row_values[["Unbound_fraction"]],
      log_Kow = row_values[["logKow"]],
      Dow = row_values[["Dow"]],
      pka = row_values[["pKa"]],
      pH = row_values[["pH"]],
      sc_blood = row_values[["sc_blood"]], 
      sc_liv = row_values[["sc_liver"]] ,
      sc_gon = row_values[["sc_gonads"]],
      sc_fat = row_values[["sc_fat"]] ,
      sc_git = row_values[["sc_GIT"]] ,
      sc_brain = row_values[["sc_brain"]] ,
      sc_kidney = row_values[["sc_kidney"]] ,
      sc_skin = row_values[["sc_skin"]] ,
      sc_rp = row_values[["sc_rpt"]],
      sc_pp = row_values[["sc_ppt"]] ,
      liv_frac = row_values[["liver_frac"]],
      gon_frac = row_values[["gonads_frac"]],
      git_frac = row_values[["GIT_frac"]],
      fat_frac = row_values[["fat_frac"]],
      brain_frac = row_values[["brain_frac"]],
      kidney_frac = row_values[["kidney_frac"]],
      skin_frac = row_values[["skin_frac"]],
      rp_frac = row_values[["rpt_frac"]],
      pp_frac = row_values[["ppt_frac"]],
      water_liver = row_values[["water_liver"]],
      water_brain = row_values[["water_brain"]],
      water_gonads = row_values[["water_gonads"]],
      water_fat = row_values[["water_fat"]],
      water_skin = row_values[["water_skin"]],
      water_GIT = row_values[["water_GIT"]],
      water_kidney = row_values[["water_kidney"]],
      water_rp = row_values[["water_rpt"]],
      water_pp = row_values[["water_ppt"]],
      lipids_liver = row_values[["lipids_liver"]],
      lipids_brain = row_values[["lipids_brain"]],
      lipids_gonads = row_values[["lipids_gonads"]],
      lipids_fat = row_values[["lipids_fat"]],
      lipids_skin = row_values[["lipids_skin"]],
      lipids_GIT = row_values[["lipids_GIT"]],
      lipids_kidney = row_values[["lipids_kidney"]],
      lipids_rp = row_values[["lipids_rpt"]],
      lipids_pp = row_values[["lipids_ppt"]]  
    )
    
    modeloutput <- output(parms)
    
    modeloutput <- as.data.frame(modeloutput)
    modeloutput_selected <- modeloutput %>%
      select(time, C_art, C_ven, C_liv, C_fat, C_gon, C_git, C_brain, C_kidney, C_skin, 
             C_rp, C_pp,  C_plasma, AA_whole_body)
    
    modeloutput_selected <- modeloutput_selected %>%
      rename(
        "Time,day"= time,
        "C_art,μg/mL"=C_art,
        "C_ven,μg/mL"=C_ven,
        "C_fat,μg/g"=C_fat,
        "C_brain,μg/g"=C_brain,
        "C_kidney,μg/g"=C_kidney,
        "C_skin,μg/g"= C_skin,
        "C_liver,μg/g" = C_liv, 
        "C_gonads,μg/g" = C_gon, 
        "C_rpt,μg/g" = C_rp, 
        "C_ppt,μg/g" = C_pp, 
        "C_plasma,μg/g"=C_plasma,
        "C_git,μg/g"=C_git,
        "C_whole_body,μg/g" = AA_whole_body)
    
    modeloutput <-modeloutput_selected
    num <- parameters_df[i, "num"]
    chemicals <- as.character(parameters_df[i, "chemicals"])
    species <- as.character(parameters_df[i, "species"])
    BW <- parameters_df[i, "BW"]
    Dose_water <- parameters_df[i, "Dose_water"]
    time_final_dose <- parameters_df[i, "time_final_dose"]
    current_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
    excel_filename <- file.path(output_dir, 
                                paste0(num, " ", species, " ", chemicals, " ", 
                                       BW, " ", time_final_dose, " ", Dose_water, 
                                       " setII_", current_time, ".xlsx"))
    write.xlsx(modeloutput, file = excel_filename,sheetName = "results", rowNames = FALSE)
  } else {
    stop(paste("One or more values are not numeric in row", i))
  }
}

