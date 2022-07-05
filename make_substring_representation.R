library(dplyr)

desired.len <- 30
step <- 5

# Load original data
mydata <- read.csv("./Ov_fullprots.csv", header = TRUE)

get_idx <- function(total.len, desired.len, step){
  startpoints <- seq(from = 1, to = (total.len - desired.len), by = step)
  startpoints <- unique(c(startpoints, total.len - desired.len))
  return(startpoints)
}

Y <- mydata %>%
  group_by(Info_protein_id) %>%
  summarise(start = get_idx(total.len = n(), desired.len, step),
            end   = start + desired.len - 1)

Xnew <- lapply(1:nrow(Y),
               function(i, Y, mydata){
                 mydata[Y$start[i]:Y$end[i], ] %>%
                   mutate(Info_strID = paste0(mydata$Info_protein_id[Y$start[i]],
                                              ":", Y$start[i]),
                          Info_index = Info_pos - Y$start[i]) %>%
                   select(Info_strID, Info_index, everything(), -Info_protein_id, -Info_pos)
               }, Y = Y, mydata = mydata) %>%
  bind_rows()

write.csv(Xnew, "./Ov_subsequences.csv", row.names = FALSE)
