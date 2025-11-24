.check_tensorflow <- function() {
  
  # nocov start
  # Present a menu/dialog to the user for confirmation
  choice <- utils::menu(
    choices = c("Yes, it is loaded", "No, it is not loaded"), 
    title = paste0(
      "-------------------------------------------------------------------\n",
      "reticulate::use_condaenv(condaenv = \"tensorflow\", required = TRUE)\n",
      "-------------------------------------------------------------------\n",
      "\n",
      "Please confirm you have loaded TensorFlow into R (enter 1 or 2)."
    )
  )
  
  # The 'menu' function returns the numeric index of the user's choice.
  # Option 1 corresponds to "Yes".
  if (choice == 1) {
    # User selected "Yes" (Option 1)
    base::message("Confirmation received. Proceeding with subsequent code...")
  } else {
    # choice is 2 (No) or 0 (Cancel/Escape)
    base::stop(
      "User selected 'No' or cancelled. Please load ", 
      "tensorflow with miniconda", 
      " and try again.", 
      call. = FALSE
    )
  }
  
  # Function returns NULL invisibly upon successful check.
  return(invisible(NULL))
  # nocov end
}
