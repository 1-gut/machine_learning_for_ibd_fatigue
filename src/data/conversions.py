def convert_urea_from_mg_dl_to_mmol_l(urea_mg_dl):
    # Convert urea from mg/dL to mmol/L
    # To convert urea concentration from mg/dL to mmol/L, you need the molar mass of urea (CO(NH₂)₂), which is approximately 60.06 g/mol.
    # The conversion formula is:
    # Concentration (mmol/L) = Concentration (mg/dL) / 6.006
    return urea_mg_dl / 6.006


def convert_creatinine_from_mg_dl_to_umol_l(creatinine_mg_dl):
    # Convert creatinine from mg/dL to µmol/L
    # To convert creatinine concentration from mg/dL to µmol/L, you need the molar mass of creatinine (C₄H₇N₃O), which is approximately 113.12 g/mol.
    # The conversion involves these steps:
    # Convert mg/dL to g/L: Multiply by 10 (since 1 dL = 0.1 L and 1 g = 1000 mg). Concentration (g/L) = Concentration (mg/dL) * (1 g / 1000 mg) / (1 dL / 10 L) = Concentration (mg/dL) * 0.01
    # Convert g/L to mol/L: Divide by the molar mass (113.12 g/mol). Concentration (mol/L) = Concentration (g/L) / 113.12 g/mol
    # Convert mol/L to µmol/L: Multiply by 1,000,000 (since 1 mol = 1,000,000 µmol). Concentration (µmol/L) = Concentration (mol/L) * 1,000,000 µmol/mol
    # Combining these steps gives the conversion factor:

    # Factor = (0.01 / 113.12) * 1,000,000 ≈ 88.4

    # Therefore, the formula used in your code is:

    # Concentration (µmol/L) = Concentration (mg/dL) * 88.4

    return creatinine_mg_dl * 88.4


def capitalize_first_letter(s):
    if isinstance(s, str) and len(s) > 0:
        return s[0].upper() + s[1:]
    return s
