#!/usr/bin/env Rscript
# Extract coords + first_type + normalised weights for the display cell types.
suppressMessages(library(SpatialExperiment))
a <- commandArgs(trailingOnly = TRUE); rds <- a[1]; out <- a[2]; sample <- a[3]
WANT <- c("DZ","LZ","FDC","GC-Tfh","B naive","B memory","B MZ","B pre")
SAFE <- c("DZ","LZ","FDC","GC_Tfh","B_naive","B_memory","B_MZ","B_pre")
cat("[", sample, "]\n")
spe <- readRDS(rds)
w <- assay(spe, "weights")
miss <- setdiff(WANT, rownames(w)); if (length(miss)) stop(paste("missing:", paste(miss, collapse=", ")))
wt <- t(as.matrix(w)); rs <- rowSums(wt); norm <- wt / rs; norm[!is.finite(norm)] <- 0
cc <- as.data.frame(spatialCoords(spe)); colnames(cc) <- c("x","y")
cd <- as.data.frame(colData(spe))
df <- data.frame(x = as.integer(cc$x), y = as.integer(cc$y),
                 spot_class = as.character(cd$spot_class),
                 first_type = as.character(cd$first_type))
for (i in seq_along(WANT)) df[[SAFE[i]]] <- round(norm[, WANT[i]], 5)
df$B_foll <- round(norm[,"B naive"] + norm[,"B memory"] + norm[,"B MZ"], 5)
cat("  spots:", nrow(df), "\n")
for (cn in c(SAFE,"B_foll")) cat("   ", cn, ">=0.15:", sum(df[[cn]] >= 0.15), "\n")
cat("    first_type in {B naive,B memory,B MZ}:",
    sum(df$first_type %in% c("B naive","B memory","B MZ")), "\n")
gz <- gzfile(out,"w"); write.csv(df, gz, row.names=FALSE); close(gz); cat("  wrote", out, "\n")
