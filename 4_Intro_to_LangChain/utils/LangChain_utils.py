from langchain.agents import tool

import sqlite3

@tool
def dict_diff(dict_1: dict, dict_2: dict) -> dict:
    """Identify value difference from dictionary 1 to dictionary 2 in python."""
    difference = {k: dict_2[k] for k in dict_2 if k in dict_1 and dict_1[k] != dict_2[k]}
    return difference
    


@tool
def query_db(name: str) -> dict:
    """Run a query on the database and return the company information"""
    conn = sqlite3.connect("./identifier.sqlite")
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM company_info WHERE navn_foretaksnavn = "{name}";')
    result = cursor.fetchone()
    columns = [col[0] for col in cursor.description]
    row_dict = dict(zip(columns, result))

    # Close the cursor
    cursor.close()
    
    # Commit any changes and close the connection
    conn.close()

    return row_dict

def query_db_all() -> dict:
    """Run a query on the database and return all info"""
    conn = sqlite3.connect("./identifier.sqlite")
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM company_info;')
    result = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    row_dict = dict(zip(columns, result))

    # Close the cursor
    cursor.close()
    
    # Commit any changes and close the connection
    conn.close()

    return result

@tool
def update_db(changed_fields: dict, name: str) -> str:
    """update the database using the given fields"""
    conn = sqlite3.connect("./identifier.sqlite")
    cursor = conn.cursor()
    set_clause = ', '.join([f"{column} = ?" for column in changed_fields.keys()])
    values = list(changed_fields.values())

    update_query = f"UPDATE company_info SET {set_clause} WHERE navn_foretaksnavn = ?;"

    values.append(name)
    cursor.execute(update_query, tuple(values))
    conn.commit()

    print('DB updated with value:', values[:-1])
    print(values)

    # Close the cursor
    cursor.close()
    
    # Commit any changes and close the connection
    conn.close()


@tool
def update_db_query_llm(update_query: str) -> str:
    """run the update query on the local db"""
    conn = sqlite3.connect("./identifier.sqlite")
    cursor = conn.cursor()
    cursor.execute(update_query)
    conn.commit()
    print('DB updated with query:', update_query)
    
    # Close the cursor
    cursor.close()
    
    # Commit any changes and close the connection
    conn.close()
    
