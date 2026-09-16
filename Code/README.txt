Minimal model reproduction package (English data version, Data-V2 + source comparison)

1. Install dependencies
   pip install -r requirements.txt

2. Run the 12 main fused-observation models
   python run_all.py

   On Windows:
   run_all.bat

3. Run the DO-Sum observation-source comparison
   python run_source_comparison.py

   This runs:
   - UAV spectral + texture
   - Sentinel-2 spectral + texture
   - Fused spectral + texture

4. Main data
   Data/Target.xlsx
   Data/1.5m-Fused.xlsx
   Data/3m-Fused.xlsx

5. Source-comparison data
   Data/UAV.xlsx
   Data/Sentinel.xlsx

   UAV.xlsx contains the fixed 126/54 Training/Verification split in English sheet format.
   Sentinel.xlsx contains both spectral and texture Training/Verification sheets.

6. Target data
   Sheets:
     SH-Training-Clean
     SH-Verification-Clean

   Sample ID:
     Sample Number

   Endpoint:
     Target

7. Predictor sheet naming
   Fused/Sentinel-2:
     BJ-Spectral-Training / BJ-Spectral-Verification
     GJ1-Spectral-Training / GJ1-Spectral-Verification
     GJ2-Spectral-Training / GJ2-Spectral-Verification
     RS-Spectral-Training / RS-Spectral-Verification
     BJ-Texture-Training / BJ-Texture-Verification
     GJ1-Texture-Training / GJ1-Texture-Verification
     GJ2-Texture-Training / GJ2-Texture-Verification
     RS-Texture-Training / RS-Texture-Verification

   UAV:
     BJ-Spectral-Training / BJ-Spectral-Verification
     GJ-Spectral-Training / GJ-Spectral-Verification
     RS-Spectral-Training / RS-Spectral-Verification
     BJ-Texture-Training / BJ-Texture-Verification
     GJ-Texture-Training / GJ-Texture-Verification
     RS-Texture-Training / RS-Texture-Verification

8. M3 final artifacts
   All M3 frozen final models and selected-feature files are stored together in:
     artifacts_M3/

   Fused:
     spectral_model.pt / spectral_features.csv
     texture_model.pt / texture_features.csv
     combined_model.pt / combined_features.csv

   Sentinel-2:
     Sentinel2_model.pt / Sentinel2_features.csv

   UAV:
     UAV_model.pt / UAV_features.csv

9. Outputs
   outputs/
