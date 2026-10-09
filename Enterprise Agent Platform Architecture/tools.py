from pydantic import BaseModel, Field

class SalaryUpdateInput(BaseModel):
    employee_id: str = Field(description="The unique employee ID, e.g., EMP-101")
    new_salary: float = Field(description="The updated salary amount in SAR")

# Metadata registry declaring sensitivity
TOOL_METADATA = {
    "get_employee_payroll": {"sensitive": False},
    "check_bank_reconciliation": {"sensitive": False},
    "update_employee_salary": {"sensitive": True} # <--- Requires Human Approval!
}

def update_employee_salary(employee_id: str, new_salary: float) -> dict:
    return {"status": "success", "message": f"Updated {employee_id} salary to {new_salary} SAR."}