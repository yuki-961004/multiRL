.theme_apa <- function (
    base_size = 12, 
    base_family = "", 
    box = FALSE
){
  adapted_theme <- ggplot2::theme_bw(base_size, base_family) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        size   = ggplot2::rel(1.1),
        margin = ggplot2::margin(0, 0, ggplot2::rel(14), 0),
        hjust  = 0.5
      ),
      plot.subtitle = ggplot2::element_text(
        size   = ggplot2::rel(0.8),
        margin = ggplot2::margin(
          ggplot2::rel(-7), 0, ggplot2::rel(14), 0
        ),
        hjust  = 0.5
      ),
      axis.title.x = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(ggplot2::rel(12), 0, 0, 0)
      ),
      axis.title.x.top = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, 0, ggplot2::rel(12), 0)
      ),
      axis.title.y = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, ggplot2::rel(12), 0, 0)
      ),
      axis.title.y.right = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, 0, 0, ggplot2::rel(12))
      ),
      axis.ticks.length = ggplot2::unit(
        ggplot2::rel(6), "points"
      ),
      axis.text = ggplot2::element_text(
        size = ggplot2::rel(0.9)
      ),
      axis.text.x = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(ggplot2::rel(6), 0, 0, 0)
      ),
      axis.text.y = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(0, ggplot2::rel(6), 0, 0)
      ),
      axis.text.y.right = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(0, 0, 0, ggplot2::rel(6))
      ),
      axis.line = ggplot2::element_line(),
      legend.title = ggplot2::element_text(),
      legend.key = ggplot2::element_rect(
        fill  = NA,
        color = NA
      ),
      legend.key.width = ggplot2::unit(
        ggplot2::rel(20), "points"
      ),
      legend.key.height = ggplot2::unit(
        ggplot2::rel(20), "points"
      ),
      legend.margin = ggplot2::margin(
        t    = ggplot2::rel(16),
        r    = ggplot2::rel(16),
        b    = ggplot2::rel(16),
        l    = ggplot2::rel(16),
        unit = "points"
      ),
      panel.spacing = ggplot2::unit(
        ggplot2::rel(14), "points"
      ),
      # 移除所有网格线
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor.y = ggplot2::element_blank(),
      # 消除 facet 标签背景
      strip.background = ggplot2::element_rect(
        fill  = NA,
        color = NA
      ),
      strip.text.x = ggplot2::element_text(
        size   = ggplot2::rel(1.2),
        margin = ggplot2::margin(0, 0, ggplot2::rel(10), 0)
      ),
      strip.text.y = ggplot2::element_text(
        size   = ggplot2::rel(1.2),
        margin = ggplot2::margin(0, 0, 0, ggplot2::rel(10))
      )
    )
  
  if (box) {
    adapted_theme <- adapted_theme +
      ggplot2::theme(
        panel.border = ggplot2::element_rect(color = "black")
      )
  } else {
    adapted_theme <- adapted_theme +
      ggplot2::theme(
        panel.border = ggplot2::element_blank()
      )
  }
  
  adapted_theme
}
