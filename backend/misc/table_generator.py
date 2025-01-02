from typing import List, Dict, Set
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_postgres_type(data_type: str) -> str:
    """Map common data types to PostgreSQL types."""
    type_mapping = {
        'integer': 'INTEGER',
        'bigint': 'BIGINT',
        'character varying': 'VARCHAR',
        'varchar': 'VARCHAR',
        'text': 'TEXT',
        'boolean': 'BOOLEAN',
        'timestamp': 'TIMESTAMP',
        'date': 'DATE',
        'numeric': 'NUMERIC',
        'float': 'FLOAT',
        'double': 'DOUBLE PRECISION',
        'json': 'JSON',
        'jsonb': 'JSONB',
        'serial': 'SERIAL PRIMARY KEY',
        'bigserial': 'BIGSERIAL PRIMARY KEY'
    }
    return type_mapping.get(data_type.lower(), 'TEXT')

def get_table_dependencies(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build a dependency graph of tables based on foreign key references."""
    dependencies = defaultdict(set)
    
    for _, row in df.iterrows():
        table_name = row['Table']
        references = row.get('References')
        
        if pd.notna(references) and references:
            referenced_table = references.split('.')[0]
            dependencies[table_name].add(referenced_table)
    
    return dependencies

def topological_sort(dependencies: Dict[str, Set[str]]) -> List[str]:
    """Sort tables in order of their dependencies."""
    visited = set()
    temp_mark = set()
    order = []
    
    def visit(table: str):
        if table in temp_mark:
            raise ValueError(f"Circular dependency detected involving table {table}")
        if table not in visited:
            temp_mark.add(table)
            for dep in dependencies.get(table, set()):
                visit(dep)
            temp_mark.remove(table)
            visited.add(table)
            order.append(table)
    
    for table in dependencies.keys():
        if table not in visited:
            visit(table)
            
    return list(reversed(order))

def create_tables_from_excel(
    file_path: str,
    connection_string: str = "postgresql://postgres:postgres@localhost:5432/postgres"
) -> None:
    """Create database tables based on schema defined in Excel file."""
    try:
        # Read Excel file
        df = pd.read_csv(file_path)
        engine = create_engine(connection_string)
        
        # Get dependencies and sort tables
        dependencies = get_table_dependencies(df)
        table_order = topological_sort(dependencies)
        
        # Add any tables that don't have dependencies
        all_tables = set(df['Table'].unique())
        for table in all_tables:
            if table not in table_order:
                table_order.append(table)
        
        with engine.connect() as connection:
            for table_name in table_order:
                table_data = df[df['Table'] == table_name]
                
                # Prepare column definitions
                columns = []
                foreign_keys = []
                
                for _, row in table_data.iterrows():
                    col_name = row['Column name']
                    data_type = get_postgres_type(row['Data type'])
                    nullable = "NULL" if row['Is Nullable'].lower() == 'yes' else "NOT NULL"
                    
                    # Handle foreign key references
                    references = row.get('References')
                    if pd.notna(references) and references:
                        ref_table, ref_column = references.split('.')
                        columns.append(f"{col_name} {data_type} {nullable}")
                        foreign_keys.append(
                            f"FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_column})"
                        )
                    else:
                        columns.append(f"{col_name} {data_type} {nullable}")
                
                # Combine column definitions and foreign key constraints
                all_definitions = columns + foreign_keys
                
                # Create table if not exists
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {','.join(all_definitions)}
                );
                """
                
                try:
                    connection.execute(text(create_table_sql))
                    connection.commit()
                    logger.info(f"Table '{table_name}' created or already exists")
                except Exception as table_error:
                    logger.error(f"Error creating table {table_name}: {table_error}")
                    connection.rollback()

    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    excel_file = "backend/misc/db_schema.csv"  # Your Excel file path
    create_tables_from_excel(excel_file)


