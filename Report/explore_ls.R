library(ggplot2)
setwd("~/kaggle-galaxy")
options(stringsAsFactors = FALSE)

# Set pdf output
f_out <- "ls_imgdim.pdf"

data <- read.csv("Submissions/submissions.csv")
ls <- data[grep("ls_", data$Name), ]
ls$Name <- factor(ls$Name, levels = unique(as.character(ls$Name)),
                  ordered = TRUE)

ggplot(data = ls, aes(x = Name, y = Kaggle, group = 1)) +
    geom_line() + geom_point(color = "red", size = 4, fill = "white") +
    ggtitle("Leaderboard Error as a function of Image Dimension") +
    xlab("Image Dimension") + ylab("Leaderboard Error")

ggsave(f_out)
