
setwd("/Users/rui.bo/Desktop/BASE/1_phd/Y3/RC_br/_json/_run/JUN11__ETHlib_3R2C_ONSITE_occ/calibro")

library(calibro)

CE <- calEnv$new(name = 'simplest')

CE$add.ds(
    name = 'data1',
    Y.star = '/Users/rui.bo/Desktop/BASE/1_phd/Y3/RC_br/_json/_run/JUN11__ETHlib_3R2C_ONSITE_occ/BC/obs.csv',
    TT     = '/Users/rui.bo/Desktop/BASE/1_phd/Y3/RC_br/_json/_run/JUN11__ETHlib_3R2C_ONSITE_occ/BC/TT.csv',
    Y      = '/Users/rui.bo/Desktop/BASE/1_phd/Y3/RC_br/_json/_run/JUN11__ETHlib_3R2C_ONSITE_occ/BC/Y.csv'
)

CE$rd = 'pca'
CE$sa = 'sobolSmthSpl'
CE$ret = list(mthd = 'ng.screening')
CE$mdls = 'gpr.ng.sePar01_whitePar01'
CE$train = list(type = 'training', alg = 'amoeba')
CE$cals = 'cal.gpr.ng'
CE$cal.mcmc = list(alg = 'amg')

CE$cal.res()

M <- CE$cal.mcmc
Z <- M$Z
# z_map <- as.data.frame(t(M$bestz))

if (inherits(Z, "mcmc.list")) {
Z <- do.call(rbind, lapply(Z, as.matrix))
} else {
Z <- as.matrix(Z)
}


write.csv(Z, "posterior_draws_Z.csv", row.names = FALSE)
# write.csv(z_map, "z_map.csv", row.names = FALSE)
theta <- CE$cal.res(mode = "theta")
write.csv(theta, "theta_physical_summary.csv", row.names = FALSE)

CE$genReport(
type = "pdf",
out = c("dss", "cal")
)
