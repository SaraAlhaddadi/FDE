import sqlite3

def get_user_data(user_input):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # Intentional SQL injection vulnerability for testing
    query = "SELECT * FROM users WHERE username = '" + user_input + "'"
    cursor.execute(query)
    return cursor.fetchall()