import sys
from src.logger import logging

class CustomException(Exception):
    """
    Custom exception class to provide detailed error information for ML pipeline.
    Inherits from the base Exception class.
    """
    def __init__(self,error_message:str,error_detail:sys):
        """
        Initialize the CustomException with detailde error information.

        Args:
            error_message: The original error message
            error_detail: System information about the error (from sys)
        """
        #Call the parent class (Exception) Constructor
        super().__init__(error_message)

        #get detailed error infromation
        self.error_message=self._generate_detailed_error_message(error_message,error_detail)

    @staticmethod
    def _generate_detailed_error_message(error_message:str,error_detail:sys) -> str:
        """
        Generate a detailed error message including file name, line number, and error description.
        Args:
            error_message: The original error message
            error_detail: system information about the error
        Returns:
            str: Formatted error message with details
        """

        #get the exception traceback details
        _,_,exc_tb=error_detail.exc_info()

        #extract file name where error occured
        file_name=exc_tb.tb_frame.f_code.co_filename

        #create detailed error message
        detailed_message= (
            f"\n→Error occured in Python script:"
            f"\n→File: {file_name}"
            f"\n→Line number:{exc_tb.tb_lineno}"
            f"\n→Error message: {str(error_message)}"
        )
        
        #log the error
        logging.error(detailed_message)
        
        return detailed_message
    
    def __str__(self) -> str:
        """
        String representation of the exception
        Returns:
            str: The detailed error message
        """
        return self.error_message