def log_message(message, verbose=False, always_print=False):
    """
    Print a formatted log message
    
    Args:
        message (str): The message to print
        verbose (bool): Whether to print detailed logs
        always_print (bool): Whether to print regardless of verbose setting
    """
    if not verbose and not always_print:
        return
    
    print(f"{message}")