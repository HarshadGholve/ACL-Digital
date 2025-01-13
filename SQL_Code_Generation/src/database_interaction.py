import psycopg2
from psycopg2 import sql

class DatabaseConnector:
    def __init__(self, db_name, user, password, host='localhost', port=5432):
        """
        Initializes a connection to the PostgreSQL database.
        :param db_name: Name of the database.
        :param user: Database user.
        :param password: Password for the database user.
        :param host: Database host (default is 'localhost').
        :param port: Database port (default is 5432).
        """
        self.connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.connection.cursor()
        print("Connected to the PostgreSQL database.")

    def execute_query(self, query, values=None):
        """
        Executes an SQL query.
        :param query: The SQL query to execute.
        :param values: Optional values to use with parameterized queries.
        :return: Query result if it's a SELECT query.
        """
        try:
            if values:
                self.cursor.execute(query, values)
            else:
                self.cursor.execute(query)

            if query.strip().upper().startswith("SELECT"):
                return self.cursor.fetchall()
            else:
                self.connection.commit()
                print("Query executed successfully.")
        except Exception as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()

    def close_connection(self):
        """
        Closes the database connection.
        """
        self.cursor.close()
        self.connection.close()
        print("Database connection closed.")

# Example usage
if __name__ == "__main__":
    # Replace with your database credentials
    db = DatabaseConnector(
        db_name="fastapi",
        user="postgres",
        password="password"
    )
    
    # Test query
    query = "SELECT * FROM users;"
    results = db.execute_query(query)
    print(results)
    
    # Close the connection
    db.close_connection()
