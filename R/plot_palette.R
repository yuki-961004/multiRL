.palette <- function(n) {
  
  # 固定调色盘（第一个灰色，后面是你的色阶）
  base_colors <- c(
    "grey",
    "#053562",  # deep blue
    "#55c186",  # green
    "#f0de36",  # yellow
    "#f79d1e",  # orange
    "#e84a34",  # red
    "#8b2f97"   # purple
  )
  
  # 如果 n <= 固定色数量：直接裁切即可
  if (n <= length(base_colors)) {
    return(base_colors[seq_len(n)])
  }
  
  # nocov start
  # 用 colorRampPalette 做渐变扩展
  dynamic_cols <- grDevices::colorRampPalette(base_colors[-1])(n - 1)
  
  # 把灰色放回第一个
  final_cols <- c(base_colors[1], dynamic_cols)
  
  return(final_cols)
  # nocov end
}
