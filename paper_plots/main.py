from harry_lee_functions import (
    compare_strfun,
    compare_exp_model_pdf_bjorken_x,
    compare_exp_model_pdf_nachtmann_xi,
    generate_table_struct_funcs,
    generate_table_xsecs,
    anl_osaka_model,
    fit_inclusive_scaling,             # global fit (already added)
    fit_inclusive_scaling_per_bin,     # individual fits
    compare_F2,                        # <-- NEW import
)

# RGA Q2 bins:  2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699

def main():
    mode = input(
        "Choose option: compare_strfun, compare_exp_model_pdf_bjorken_x_nachtmann_xi, "
        "compare_F2, generate_table, generate_xsecs, fit_inclusive_scaling, "
        "fit_inclusive_scaling_individual : "
    ).strip().lower().strip("'\"")

    valid_modes = {
        "compare_strfun",
        "compare_exp_model_pdf_bjorken_x_nachtmann_xi",
        "compare_f2",                           # <-- NEW mode
        "generate_table",
        "generate_xsecs",
        "fit_inclusive_scaling",                # global
        "fit_inclusive_scaling_individual",     # per-Q2
    }

    if mode not in valid_modes:
        print("Supported options: compare_strfun, compare_exp_model_pdf_bjorken_x_nachtmann_xi, "
              "compare_F2, generate_table, generate_xsecs, fit_inclusive_scaling, "
              "fit_inclusive_scaling_individual")
        return

    # Fit modes: no user-provided Q²/beam energy
    if mode in {"fit_inclusive_scaling", "fit_inclusive_scaling_individual"}:
        try:
            if mode == "fit_inclusive_scaling":
                print("\n→ Running global scaling fit over exp_data_to_fit/ …")
                fit_inclusive_scaling()
            else:
                print("\n→ Running per-Q² (individual) scaling fits over exp_data_to_fit/ …")
                fit_inclusive_scaling_per_bin()
        except Exception as err:
            print(f"  Error: {err}")
        return

    # Ask for Q² values for the remaining modes
    try:
        q2_vals = [
            float(token) for token in
            input("Enter Q² values (GeV²) separated by commas: ").split(",")
            if token.strip()
        ]
    except ValueError:
        print("Please enter only numbers separated by commas.")
        return

    # compare_F2 does NOT need beam energy; call once and return
    if mode == "compare_f2":
        try:
            print(f"\n→ Plotting F2 for Q² list: {q2_vals}")
            compare_F2(q2_vals)
        except Exception as err:
            print(f"  Error: {err}")
        return

    # Modes that require beam energy
    if mode in {"compare_strfun", "compare_exp_model_pdf_bjorken_x_nachtmann_xi", "generate_xsecs"}:
        try:
            beam_energy = float(input("Enter beam energy E (GeV): "))
        except ValueError:
            print("Please enter a valid beam energy.")
            return

    # Process per-Q² for the remaining modes
    for q2 in q2_vals:
        print(f"\n→ Processing Q² = {q2} GeV² …")
        try:
            if mode == "compare_strfun":
                compare_strfun(q2, beam_energy)
            elif mode == "compare_exp_model_pdf_bjorken_x_nachtmann_xi":
                compare_exp_model_pdf_bjorken_x(q2, beam_energy)
                compare_exp_model_pdf_nachtmann_xi(q2, beam_energy)
            elif mode == "generate_table":
                generate_table_struct_funcs(file_path="input_data/wempx.dat", fixed_Q2=q2)
            elif mode == "generate_xsecs":
                generate_table_xsecs(file_path="input_data/wempx.dat", fixed_Q2=q2, beam_energy=beam_energy)
        except Exception as err:
            print(f"  Error: {err}")


if __name__ == "__main__":
    main()
