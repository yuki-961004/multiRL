.detect_data <- function(data) {
  # 1. Find column names containing "sub" 
  # (case-insensitive)
  col_names <- names(data)
  subject_col_candidates <- col_names[grepl("sub", col_names, ignore.case = TRUE)]
  
  if (length(subject_col_candidates) == 0) {
    stop(
      "Error: Could not find a subject ID column containing 'sub' in the data. 
      Please check the column names."
    )
  }
  
  # Default to the first one found
  subject_col_name <- subject_col_candidates[1] 
  if (length(subject_col_candidates) > 1) {
    message(paste(
      "Found multiple columns containing 'sub': ",
      paste(subject_col_candidates, collapse = ", "),
      ". Using the first one found: '", subject_col_name, "'", sep = ""
    ))
  } else {
    message(paste(
      "Automatically detected subject ID column as: '", 
      subject_col_name, 
      "'", sep = ""
    ))
  }
  
  # 2. Randomly select an existing subject ID from the column 
  # (ensure NA values are excluded)
  # First, get all IDs from the column, then remove NAs
  all_ids_in_column <- data[[subject_col_name]]
  valid_subject_ids <- unique(all_ids_in_column[!is.na(all_ids_in_column)])
  
  if (length(valid_subject_ids) == 0) {
    stop(paste(
      "Error: No valid subject IDs found in column '", 
      subject_col_name, 
      "' (column might only contain NAs, be empty, 
      or all values became NA after processing).", 
      sep = ""
    ))
  }
  
  random_subject_id <- sample(valid_subject_ids, 1)
  
  # 构建包含四个命名元素的列表
  res <- list(
    # 第1个元素: subject那一列的列名 (字符串)
    sub_col_name = subject_col_name,
    # 第2个元素: 随机选取的ID (尝试转为数值型)
    random_id = as.numeric(random_subject_id),
    # 第3个元素: subject列所有存在的ID向量
    all_ids = valid_subject_ids
  )
  
  return(res)
}
