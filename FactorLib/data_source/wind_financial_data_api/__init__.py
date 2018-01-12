from .database import (WindIncomeSheet,
                       WindEstDB
                       )

incomesheet = WindIncomeSheet()
estsheet = WindEstDB()

__all__ = ['incomesheet', 'estsheet']