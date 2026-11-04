"""Tests run on the code's defaults, not on this laptop's .env (its booth settings, such as AUTO_PLACE=false, broke the place tests)."""
import os

os.environ["PUPPY_NO_DOTENV"] = "1"
