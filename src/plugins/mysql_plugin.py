# uv run python -m src.plugins.mysql_plugin
# NOTE: This plugin requires the 'mysql-connector-python' package.
# You can install it using: pip install mysql-connector-python

import os
import time
from typing import Any, List, Tuple, Dict

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import errorcode

from src.plugins.base import DBConnector
from src.plugins.manager import PluginManager


class MySQLConnectionManager:
    """
    A singleton class for managing MySQL server connections.
    
    This manager handles connection setup and maintenance using credentials 
    configured via environment variables.
    
    Required Environment Variables:
        MYSQL_USER: Username for MySQL authentication
        MYSQL_PASSWORD: Password for MySQL authentication
        MYSQL_DATABASE: Target database name
        MYSQL_HOST: (Optional) Server hostname (default: localhost)
        MYSQL_PORT: (Optional) Server port (default: 3306)
    """
    load_dotenv()
    _instance = None
    _connection = None

    def __new__(cls):
        """
        Create a new instance of the singleton, or return the existing one.
        """
        if cls._instance is None:
            cls._instance = super(MySQLConnectionManager, cls).__new__(cls)
        return cls._instance

    def get_connection(self) -> mysql.connector.MySQLConnection:
        """
        Get the MySQL connection, establishing it if it doesn't exist or is closed.

        Returns:
            mysql.connector.MySQLConnection: The active MySQL connection object.
        """
        if self._connection is None or not self._connection.is_connected():
            self._connection = self._establish_mysql_connection()
        return self._connection

    def _establish_mysql_connection(self) -> mysql.connector.MySQLConnection:
        """
        Establish a connection to MySQL using environment variables.

        Returns:
            mysql.connector.MySQLConnection: A new MySQL connection object.
            
        Raises:
            ValueError: If required environment variables are missing.
            ConnectionError: If the connection fails for any reason.
        """
        config = {
            'host': os.environ.get('MYSQL_HOST', 'localhost'),
            'user': os.environ.get('MYSQL_USER'),
            'password': os.environ.get('MYSQL_PASSWORD'),
            'database': os.environ.get('MYSQL_DATABASE'),
            'port': int(os.environ.get('MYSQL_PORT', '3306'))
        }

        missing_vars = [key.upper() for key, value in config.items() if value is None and key in ['user', 'password', 'database']]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        try:
            connection = mysql.connector.connect(**config)
            return connection
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise ConnectionError("Authentication failed: Invalid user name or password.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                raise ConnectionError(f"Database '{config['database']}' does not exist.")
            else:
                raise ConnectionError(f"Failed to establish MySQL connection: {err}") from err


class MySQLConnector(DBConnector):
    """
    MySQLConnector is a plugin that interacts with a MySQL database.
    It implements the common DBConnector interface with connection state management.
    """
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initializes the MySQLConnector.
        
        Args:
            max_retries (int): Maximum number of retry attempts for queries.
            retry_delay (float): Delay in seconds between retries.
        """
        self.connection = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to the database.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._connected and self.connection is not None and self.connection.is_connected()

    def connect(self) -> Tuple[mysql.connector.MySQLConnection, str]:
        """
        Establish the connection using the MySQLConnectionManager singleton.
        
        Returns:
            tuple: A tuple containing the connection object and the database name.
            
        Raises:
            ConnectionError: If the connection cannot be established.
        """
        try:
            self.connection = MySQLConnectionManager().get_connection()
            self._connected = True
            return self.connection, self.connection.database
        except Exception as e:
            self._connected = False
            print(f"Connection failed: {type(e).__name__}: {str(e)}")
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}") from e

    def execute_query(self, query: str) -> Tuple[List[Tuple], Any]:
        """
        Execute a query against the MySQL database with retry logic.
        
        Args:
            query (str): A SQL query string.
            
        Returns:
            tuple: A tuple containing the fetched rows and the cursor description.
            
        Raises:
            ConnectionError: If the query fails after all retry attempts.
        """
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                if not self.is_connected:
                    self.connect()
                
                cursor = self.connection.cursor()
                cursor.execute(query)

                if cursor.description is None:  # For INSERT, UPDATE, DELETE
                    self.connection.commit()
                    cursor.close()
                    return [], None

                rows = cursor.fetchall()
                description = cursor.description
                cursor.close()
                return rows, description

            except mysql.connector.Error as err:
                last_error = err
                attempts += 1
                self._connected = False 
                print(f"Attempt {attempts} failed: {type(err).__name__}: {str(err)}")
                
                if err.errno in (errorcode.ER_ACCESS_DENIED_ERROR, errorcode.ER_BAD_DB_ERROR):
                    raise ConnectionError(f"Query failed with a non-recoverable error: {str(last_error)}") from last_error

                if attempts < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                attempts += 1
                self._connected = False
                print(f"Attempt {attempts} failed: {type(e).__name__}: {str(e)}")
                if attempts < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Query failed after {self.max_retries} attempts. Last error: {str(last_error)}")

    def disconnect(self) -> None:
        """
        Close the database connection and clean up resources.
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
        self.connection = None
        self._connected = False


def dataframe_from_mysql_response(response: Tuple[List[Tuple], Any]) -> pd.DataFrame:
    """
    Convert a MySQL response tuple into a pandas DataFrame.
    
    Args:
        response (tuple): A tuple containing a list of rows and the cursor description.
        
    Returns:
        pd.DataFrame: A DataFrame containing the query results.
    """
    rows, description = response
    if not rows or not description:
        return pd.DataFrame()
    
    columns = [desc[0] for desc in description]
    return pd.DataFrame(rows, columns=columns)



# uv run python -m src.plugins.mysql_plugin
if __name__ == "__main__":
    PluginManager.register_plugin("mysql", MySQLConnector)
    mysql_plugin = None
    try:
        mysql_plugin = PluginManager.get_plugin("mysql")
        mysql_plugin.connect()
        print("Successfully connected to MySQL.")
        
        query = """
        SELECT * FROM control_minutedata 
        WHERE controlId = '2916276' 
        AND locationId = '11688'
        AND timeStamp BETWEEN '2025-02-05' AND '2025-05-05'
        """
        response = mysql_plugin.execute_query(query)
        
        df = dataframe_from_mysql_response(response)
        print("\nTables in the database:")
        print(df)
        
    except (ValueError, ConnectionError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
    finally:
        if mysql_plugin and mysql_plugin.is_connected:
            mysql_plugin.disconnect()
            print("\nDisconnected from MySQL.")
