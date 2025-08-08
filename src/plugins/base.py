from abc import ABC, abstractmethod
from typing import Any, Tuple

class DBConnector(ABC):
    """
    Abstract base class defining the interface for database connectors.
    All database connector plugins must implement these methods.
    """
    
    @abstractmethod
    def connect(self) -> Tuple[Any, Any]:
        """
        Establish a connection to the database.
        
        Returns:
            tuple: Connection details specific to the database implementation
            
        Raises:
            ConnectionError: If connection cannot be established
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def execute_query(self, query: str) -> Any:
        """
        Execute a query against the database and return the results.
        
        Args:
            query (str): The query string to execute
            
        Returns:
            Any: Query results in implementation-specific format
            
        Raises:
            ConnectionError: If connection is lost during query execution
            ValueError: If query is invalid
        """
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection and clean up resources.
        
        Raises:
            ConnectionError: If disconnection fails
        """
        pass

    def __enter__(self):
        """Enable context manager support"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources"""
        self.disconnect()