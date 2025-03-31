import os
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import base64
from io import BytesIO
from fastapi.staticfiles import StaticFiles
import random
import re

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root URL
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

# Global variables
uploaded_csv_path = "uploaded_data.csv"
df = None
excel_dfs = {}
chat_history: List[dict] = []

# Set OpenAI API key (use environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-actual-openai-api-key")  # Replace fallback with your key for local testing

# Pydantic models
class QueryModel(BaseModel):
    question: str

class AggregateModel(BaseModel):
    operation: str
    aggregate_type: str = "sum"

class TypeChangeModel(BaseModel):
    column: str
    new_type: str

class VlookupModel(BaseModel):
    sheet1: str
    sheet2: str
    key_column: str
    value_column: str
    new_column_name: str

# Utility function for random colors
def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Upload endpoint
@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...), sheet_name: str = Query(None)):
    global df, excel_dfs, chat_history
    chat_history = []
    filename = file.filename.lower()
    
    with open(uploaded_csv_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_csv_path)
            excel_dfs = {"main": df}
            message = f"Uploaded CSV file '{filename}'"
        elif filename.endswith('.xlsx'):
            excel_dfs = pd.read_excel(uploaded_csv_path, sheet_name=None)
            if sheet_name and sheet_name in excel_dfs:
                df = excel_dfs[sheet_name]
                message = f"Uploaded Excel file '{filename}', selected sheet '{sheet_name}'"
            else:
                df = list(excel_dfs.values())[0]
                sheet_names = list(excel_dfs.keys())
                message = f"Uploaded Excel file '{filename}', {len(sheet_names)} sheets found: {', '.join(sheet_names)}. Using first sheet '{sheet_names[0]}' by default."
        else:
            error_msg = f"Unsupported file type: '{filename}'. Please upload a .csv or .xlsx file."
            chat_history.append({"question": "File upload", "answer": error_msg})
            return {"error": error_msg}
        
        chat_history.append({"question": "File upload", "answer": message})
        return {
            "message": message,
            "sheets": list(excel_dfs.keys()),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        error_msg = f"Failed to read file '{filename}': {str(e)}"
        chat_history.append({"question": "File upload", "answer": error_msg})
        return {"error": error_msg}

# Aggregate endpoint
@app.post("/aggregate/")
async def perform_aggregate(agg: AggregateModel):
    global chat_history
    if df is None:
        chat_history.append({"question": f"Aggregate: {agg.operation}", "answer": "No file uploaded yet."})
        return {"error": "No file uploaded yet."}
    
    try:
        operation = agg.operation.strip().upper()
        result_col = None
        if " AS " in operation:
            operation, result_col = operation.split(" AS ")
            result_col = result_col.strip()

        func_match = re.match(r"(\w+)\((.*?)\)", operation)
        if func_match:
            func_name, args_str = func_match.groups()
            args = [arg.strip() for arg in args_str.split(",")]

            # Mathematical & Trigonometric
            if func_name == "SUM":
                result = df[args[0]].sum() if len(args) == 1 else df[args].sum().sum()
            elif func_name == "AVERAGE":
                result = df[args[0]].mean() if len(args) == 1 else df[args].mean().mean()
            elif func_name == "ROUND":
                result = df[args[0]].round(int(args[1]))
            elif func_name == "INT":
                result = df[args[0]].astype(int)
            elif func_name == "MOD":
                result = df[args[0]] % float(args[1])
            elif func_name == "RAND":
                result = pd.Series(np.random.rand(len(df)))
            elif func_name == "PI":
                result = pd.Series(np.pi, index=df.index)
            elif func_name == "SQRT":
                result = np.sqrt(df[args[0]])
            # Statistical
            elif func_name == "COUNT":
                result = df[args[0]].count()
            elif func_name == "COUNTA":
                result = df[args[0]].notna().sum()
            elif func_name == "COUNTIF":
                condition = args[1]
                result = df[args[0]].apply(lambda x: eval(f"x {condition}")).sum()
            elif func_name == "MAX":
                result = df[args[0]].max()
            elif func_name == "MIN":
                result = df[args[0]].min()
            elif func_name == "MEDIAN":
                result = df[args[0]].median()
            elif func_name == "MODE":
                result = df[args[0]].mode()[0]
            # Logical
            elif func_name == "IF":
                condition = args[0]
                true_val = args[1]
                false_val = args[2]
                col_name, cond = condition.split(" ", 1)
                result = np.where(df[col_name].apply(lambda x: eval(f"x {cond}")), true_val, false_val)
            elif func_name == "AND":
                result = df[args].apply(lambda row: all(eval(f"row['{arg}'] > 0") for arg in args), axis=1)
            elif func_name == "OR":
                result = df[args].apply(lambda row: any(eval(f"row['{arg}'] > 0") for arg in args), axis=1)
            elif func_name == "NOT":
                result = ~df[args[0]].astype(bool)
            # Lookup & Reference
            elif func_name == "VLOOKUP":
                key_col, table_col, index = args[0], args[1], int(args[2])
                result = df[key_col].map(df[table_col].shift(-index))
            elif func_name == "INDEX":
                row_num = int(args[1])
                col = args[0]
                result = df[col].iloc[row_num]
            elif func_name == "MATCH":
                value = args[0]
                col = args[1]
                result = df[col].index[df[col] == value].tolist()[0]
            # Text
            elif func_name == "LEFT":
                result = df[args[0]].str[:int(args[1])]
            elif func_name == "RIGHT":
                result = df[args[0]].str[-int(args[1]):]
            elif func_name == "MID":
                start, num = int(args[1]), int(args[2])
                result = df[args[0]].str[start:start+num]
            elif func_name == "LEN":
                result = df[args[0]].str.len()
            elif func_name == "CONCATENATE":
                result = df[args].agg(''.join, axis=1)
            elif func_name == "TRIM":
                result = df[args[0]].str.strip()
            elif func_name == "PROPER":
                result = df[args[0]].str.title()
            elif func_name == "LOWER":
                result = df[args[0]].str.lower()
            elif func_name == "UPPER":
                result = df[args[0]].str.upper()
            elif func_name == "SUBSTITUTE":
                result = df[args[0]].str.replace(args[1], args[2])
            # Date & Time
            elif func_name == "TODAY":
                result = pd.Series(pd.Timestamp.today().date(), index=df.index)
            elif func_name == "NOW":
                result = pd.Series(pd.Timestamp.now(), index=df.index)
            elif func_name == "YEAR":
                result = pd.to_datetime(df[args[0]]).dt.year
            elif func_name == "MONTH":
                result = pd.to_datetime(df[args[0]]).dt.month
            elif func_name == "DAY":
                result = pd.to_datetime(df[args[0]]).dt.day
            elif func_name == "HOUR":
                result = pd.to_datetime(df[args[0]]).dt.hour
            elif func_name == "MINUTE":
                result = pd.to_datetime(df[args[0]]).dt.minute
            elif func_name == "SECOND":
                result = pd.to_datetime(df[args[0]]).dt.second
            elif func_name == "DATEDIF":
                start, end, unit = args[0], args[1], args[2]
                diff = pd.to_datetime(df[end]) - pd.to_datetime(df[start])
                if unit == "D":
                    result = diff.dt.days
                elif unit == "M":
                    result = diff.dt.days / 30
                elif unit == "Y":
                    result = diff.dt.days / 365
            # Financial
            elif func_name == "PV":
                rate, nper, pmt = float(args[0]), int(args[1]), float(args[2])
                result = pmt * (1 - (1 + rate) ** -nper) / rate
            elif func_name == "FV":
                rate, nper, pmt = float(args[0]), int(args[1]), float(args[2])
                result = pmt * (((1 + rate) ** nper) - 1) / rate
            elif func_name == "PMT":
                rate, nper, pv = float(args[0]), int(args[1]), float(args[2])
                result = pv * rate / (1 - (1 + rate) ** -nper)
            else:
                result = df.eval(operation.lower())

            if agg.aggregate_type != "none":
                agg_func = {
                    "sum": result.sum() if isinstance(result, pd.Series) else result,
                    "mean": result.mean() if isinstance(result, pd.Series) else result,
                    "max": result.max() if isinstance(result, pd.Series) else result,
                    "min": result.min() if isinstance(result, pd.Series) else result,
                    "count": result.count() if isinstance(result, pd.Series) else 1
                }
                agg_result = agg_func.get(agg.aggregate_type, "Invalid aggregate type")
                message = f"{func_name} result ({agg.aggregate_type}): {agg_result}"
            else:
                agg_result = result
                message = f"{func_name} result computed"

            if result_col:
                df[result_col] = result
                message = f"New column '{result_col}' created with {func_name} result"
            
            chat_history.append({"question": f"Aggregate: {agg.operation}, {agg.aggregate_type}", "answer": message})
            return {"message": message, "result": str(agg_result) if agg.aggregate_type != "none" else None}
        
        result = df.eval(operation.lower())
        if agg.aggregate_type != "none":
            agg_func = {"sum": result.sum(), "mean": result.mean(), "max": result.max(), "min": result.min(), "count": result.count()}
            agg_result = agg_func.get(agg.aggregate_type, "Invalid aggregate type")
            message = f"Aggregate result ({agg.aggregate_type}): {agg_result}"
        else:
            agg_result = result
            message = "Operation computed"

        if result_col:
            df[result_col] = result
            message = f"New column '{result_col}' created"

        chat_history.append({"question": f"Aggregate: {agg.operation}, {agg.aggregate_type}", "answer": message})
        return {"message": message, "result": str(agg_result) if agg.aggregate_type != "none" else None}
    except Exception as e:
        error_msg = f"Failed to perform operation: {str(e)}"
        chat_history.append({"question": f"Aggregate: {agg.operation}", "answer": error_msg})
        return {"error": error_msg}

# Change type endpoint
@app.post("/change-type/")
async def change_column_type(type_change: TypeChangeModel):
    global chat_history
    if df is None:
        chat_history.append({"question": f"Change type: {type_change.column}", "answer": "No file uploaded yet."})
        return {"error": "No file uploaded yet."}
    
    try:
        col = type_change.column
        if col not in df.columns:
            error_msg = f"Column '{col}' not found."
            chat_history.append({"question": f"Change type: {col}", "answer": error_msg ()
            return {"error": error_msg}
        
        if type_change.new_type == "float":
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif type_change.new_type == "double":
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        else:
            error_msg = "Supported types: 'float', 'double'"
            chat_history.append({"question": f"Change type: {col}", "answer": error_msg})
            return {"error": error_msg}
        
        message = f"Column '{col}' changed to {type_change.new_type}"
        chat_history.append({"question": f"Change type: {col} to {type_change.new_type}", "answer": message})
        return {"message": message}
    except Exception as e:
        error_msg = f"Failed to change type: {str(e)}"
        chat_history.append({"question": f"Change type: {type_change.column}", "answer": error_msg})
        return {"error": error_msg}

# VLOOKUP endpoint
@app.post("/vlookup/")
async def perform_vlookup(vlookup: VlookupModel):
    global chat_history
    if not excel_dfs:
        chat_history.append({"question": f"VLOOKUP: {vlookup.sheet1}", "answer": "No Excel file uploaded yet."})
        return {"error": "No Excel file uploaded yet."}
    
    if vlookup.sheet1 not in excel_dfs or vlookup.sheet2 not in excel_dfs:
        error_msg = "Invalid sheet names."
        chat_history.append({"question": f"VLOOKUP: {vlookup.sheet1} with {vlookup.sheet2}", "answer": error_msg})
        return {"error": error_msg}
    
    try:
        sheet1_df = excel_dfs[vlookup.sheet1]
        sheet2_df = excel_dfs[vlookup.sheet2]
        
        merged_df = sheet1_df.merge(
            sheet2_df[[vlookup.key_column, vlookup.value_column]],
            on=vlookup.key_column,
            how='left'
        )
        
        df[vlookup.new_column_name] = merged_df[vlookup.value_column]
        excel_dfs[vlookup.sheet1] = merged_df
        
        message = f"VLOOKUP completed. New column '{vlookup.new_column_name}' added to {vlookup.sheet1}"
        chat_history.append({"question": f"VLOOKUP: {vlookup.sheet1} with {vlookup.sheet2}", "answer": message})
        return {"message": message}
    except Exception as e:
        error_msg = f"Failed to perform VLOOKUP: {str(e)}"
        chat_history.append({"question": f"VLOOKUP: {vlookup.sheet1}", "answer": error_msg})
        return {"error": error_msg}

# Ask endpoint
@app.post("/ask/")
async def ask_question(query: QueryModel):
    global chat_history
    if df is None:
        chat_history.append({"question": query.question, "answer": "No file uploaded yet."})
        return {"error": "No file uploaded yet."}
    
    question = query.question.lower()
    words = question.split()

    if "last updated" in question:
        timestamp = os.path.getmtime(uploaded_csv_path)
        answer = f"The file was last updated on {datetime.fromtimestamp(timestamp)}"
    elif "columns" in question:
        answer = f"Columns: {', '.join(df.columns)}"
    elif "summary" in question:
        answer = str(df.describe())
    elif "rows" in question:
        answer = f"Total rows: {len(df)}"
    else:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in question:
                max_val = df[col].max()
                min_val = df[col].min()
                sum_val = df[col].sum()
                mean_val = df[col].mean()
                count_val = df[col].count()
                distinct_values = df[col].unique().tolist()

                count_of_max = (df[col] == max_val).sum()
                count_of_min = (df[col] == min_val).sum()
                sum_of_max = count_of_max * max_val
                sum_of_min = count_of_min * min_val
                mean_of_max = sum_of_max / count_of_max if count_of_max != 0 else 0
                mean_of_min = sum_of_min / count_of_min if count_of_min != 0 else 0

                if "distinct" in question:
                    answer = f"Distinct values of {col}: {distinct_values}"
                elif "count of max" in question:
                    answer = f"Count of maximum {col}: {count_of_max}"
                elif "count of min" in question:
                    answer = f"Count of minimum {col}: {count_of_min}"
                elif "sum of max" in question:
                    answer = f"Sum of maximum {col}: {sum_of_max}"
                elif "sum of min" in question:
                    answer = f"Sum of minimum {col}: {sum_of_min}"
                elif "mean of max" in question:
                    answer = f"Mean of maximum {col}: {mean_of_max}"
                elif "mean of min" in question:
                    answer = f"Mean of minimum {col}: {mean_of_min}"
                elif "maximum" in words or "max" in words:
                    answer = f"Maximum {col}: {max_val}"
                elif "minimum" in words or "min" in words:
                    answer = f"Minimum {col}: {min_val}"
                elif "sum" in words:
                    answer = f"Sum of {col}: {sum_val}"
                elif "average" in words or "mean" in words:
                    answer = f"Mean {col}: {mean_val}"
                elif "count" in words:
                    answer = f"Count of {col}: {count_val}"
                break
        else:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a CSV data expert."},
                        {"role": "user", "content": f"The dataset columns are {', '.join(df.columns)}. Answer the question: {question}"}
                    ]
                )
                answer = response['choices'][0]['message']['content']
            except Exception as e:
                answer = f"Failed to process query with OpenAI: {str(e)}"
    
    chat_history.append({"question": query.question, "answer": answer})
    return {"answer": answer}

# Chat history endpoints
@app.get("/chat-history/")
async def get_chat_history():
    return {"chat_history": chat_history}

@app.post("/clear-history/")
async def clear_chat_history():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared."}

# Plot endpoint
@app.get("/plot/")
async def plot_graph(x: str = Query(...), y: str = Query(None), plot_type: str = Query("scatter"), y_columns: List[str] = Query(None)):
    global chat_history
    if df is None:
        chat_history.append({"question": f"Plot {plot_type}: {x}", "answer": "No file uploaded yet."})
        return {"error": "No file uploaded yet."}
    
    y_cols = y_columns if y_columns else ([y] if y else [])
    all_cols = [x] + y_cols
    invalid_cols = [col for col in all_cols if col and col not in df.columns]
    if invalid_cols:
        chat_history.append({"question": f"Plot {plot_type}: {x} vs {', '.join(y_cols)}", "answer": f"Invalid column names: {', '.join(invalid_cols)}"})
        return {"error": f"Invalid column names: {', '.join(invalid_cols)}"}
    
    if plot_type in ["scatter", "line", "bar"] and not y_cols:
        chat_history.append({"question": f"Plot {plot_type}: {x}", "answer": f"{plot_type.capitalize()} plot requires at least one Y-column."})
        return {"error": f"{plot_type.capitalize()} plot requires at least one Y-column."}
    
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == "scatter" and y_cols:
            for y_col in y_cols:
                sns.scatterplot(x=df[x], y=df[y_col], color=random_color(), label=y_col)
            plt.legend()
        elif plot_type == "line" and y_cols:
            for y_col in y_cols:
                sns.lineplot(x=df[x], y=df[y_col], color=random_color(), label=y_col)
            plt.legend()
        elif plot_type == "bar" and y_cols:
            df_melted = df.melt(id_vars=[x], value_vars=y_cols, var_name="Category", value_name="Value")
            sns.barplot(x=x, y="Value", hue="Category", data=df_melted, palette=[random_color() for _ in y_cols])
            plt.legend(title="Columns")
        elif plot_type == "pie":
            if y_cols:
                y_col = y_cols[0]
                plt.pie(df[y_col].value_counts(), labels=df[y_col].value_counts().index, autopct='%1.1f%%', colors=[random_color() for _ in df[y_col].unique()])
            else:
                plt.pie(df[x].value_counts(), labels=df[x].value_counts().index, autopct='%1.1f%%', colors=[random_color() for _ in df[x].unique()])
        elif plot_type == "histogram" and y_cols:
            for y_col in y_cols:
                sns.histplot(df[y_col], bins=20, color=random_color(), label=y_col, alpha=0.6)
            plt.legend()
        elif plot_type == "histogram" and not y_cols:
            sns.histplot(df[x], bins=20, color=random_color())
        else:
            chat_history.append({"question": f"Plot {plot_type}: {x} vs {', '.join(y_cols)}", "answer": "Invalid plot type."})
            return {"error": "Invalid plot type. Supported: scatter, line, bar, pie, histogram"}
        
        plt.xlabel(x)
        if y_cols and plot_type != "pie":
            plt.ylabel("Values")
        plt.title(f"{plot_type.capitalize()} of {x} vs {', '.join(y_cols) if y_cols else x}")
        
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        chat_history.append({
            "question": f"Plot {plot_type}: {x} vs {', '.join(y_cols) if y_cols else x}",
            "answer": "Graph generated",
            "image": f"data:image/png;base64,{image_base64}"
        })
        return {"image": f"data:image/png;base64,{image_base64}"}
    except Exception as e:
        error_msg = f"Failed to generate plot: {str(e)}"
        chat_history.append({"question": f"Plot {plot_type}: {x} vs {', '.join(y_cols)}", "answer": error_msg})
        return {"error": error_msg}

# Download endpoint
@app.get("/download/")
async def download_file(file_type: str = Query("csv")):
    global chat_history
    if df is None:
        chat_history.append({"question": f"Download {file_type}", "answer": "No file uploaded yet."})
        return {"error": "No file uploaded yet."}
    
    if file_type == "csv":
        output_path = "modified_data.csv"
        df.to_csv(output_path, index=False)
    elif file_type == "excel":
        output_path = "modified_data.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, sheet_df in excel_dfs.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        chat_history.append({"question": f"Download {file_type}", "answer": "Supported file types: 'csv', 'excel'"})
        return {"error": "Supported file types: 'csv', 'excel'"}
    
    chat_history.append({"question": f"Download {file_type}", "answer": f"{file_type.upper()} file prepared for download"})
    return FileResponse(output_path, filename=f"modified_data.{file_type}", media_type=f"application/{'vnd.openxmlformats-officedocument.spreadsheetml.sheet' if file_type == 'excel' else 'octet-stream'}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")