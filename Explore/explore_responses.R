library(ggplot2)
library(reshape2)
setwd("~/kaggle-galaxy/")
sol_dir <- "Data/train_solutions.csv"
options(stringsAsFactors = FALSE)

# Read in solutions matrix
f_out <- "Explore/responses_"
sol <- read.csv(sol_dir, header = TRUE)

# Plot all probabilities
x <- melt(sol, id.vars = "GalaxyID")
qplot(value, data = x, binwidth = 0.02) +
    ggtitle("Distribution of raw respones (mean = 0.14, median = 0.008)") +
    xlab("response")
ggsave(paste0(f_out, "1.pdf"), height = 6, width = 6)

