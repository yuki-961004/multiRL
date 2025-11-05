.get_dfunc <- function(func) {
  # 把函数体拼成单一字符串
  txt <- base::paste(base::deparse(base::body(func)), collapse = "\n")
  
  # 匹配形如 stats::dbeta 或 dbeta 的调用，并捕获 d 之后的分布名
  pattern <- "(?:stats::)?d([A-Za-z][A-Za-z0-9_.]*)\\s*\\("
  
  # 在单字符串上查找第一个匹配
  m <- base::regexpr(pattern, txt, perl = TRUE)
  
  # 未找到则返回 NA
  if (m[1] == -1) return(NA_character_)
  
  # 提取完整匹配串（例如 "stats::dbeta(" 或 "dbeta("）
  matched <- base::regmatches(txt, m)
  
  # 用 sub 提取第一个捕获组（分布名）
  dist <- base::sub("(?:stats::)?d([A-Za-z][A-Za-z0-9_.]*)\\s*\\(.*", "\\1", matched, perl = TRUE)
  
  return(dist)
}
