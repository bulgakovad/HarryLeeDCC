from harry_lee_functions import (
    compare_strfun,
    compare_exp_model_pdf_bjorken_x,
    compare_exp_model_pdf_nachtmann_xi,
    generate_table_struct_funcs,
    generate_table_xsecs,
    generate_pseudodata,
    compare_f2_tmc
)

#1.025,2.025,3.025,4.025,5,6.065,7.093,8.294,9.699

def main():
    mode = input(
        "Choose option: compare_strfun, compare_exp_model_pdf_bjorken_x_nachtmann_xi, "
        "generate_table, generate_xsecs, generate_pseudodata, compare_f2_tmc"
    ).strip().lower().strip("'\"")  # Strip any quotes from user input

    valid_modes = {
        "compare_strfun",
        "compare_exp_model_pdf_bjorken_x_nachtmann_xi",
        "generate_table",
        "generate_xsecs",
        "generate_pseudodata",
        "compare_f2_tmc"
    }

    if mode not in valid_modes:
        print("Supported options: compare_strfun, compare_exp_model_pdf_bjorken_x_nachtmann_xi, "
              "generate_table, generate_xsecs, generate_pseudodata, compare_f2_tmc.")
        return

    try:
        q2_vals = [
            float(token) for token in
            input("Enter Q² values (GeV²) separated by commas: ").split(",")
            if token.strip()
        ]
    except ValueError:
        print("Please enter only numbers separated by commas.")
        return

    if mode in {"compare_strfun", "compare_exp_model_pdf_bjorken_x_nachtmann_xi", "generate_xsecs", "generate_pseudodata"}:
        try:
            beam_energy = float(input("Enter beam energy E (GeV): "))
        except ValueError:
            print("Please enter a valid beam energy.")
            return

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
            elif mode == "generate_pseudodata":
                generate_pseudodata(q2_vals, beam_energy)
        except Exception as err:
            print(f"  Error: {err}")

    if mode == "compare_f2_tmc":
        compare_f2_tmc(q2_vals)


if __name__ == "__main__":
    main()
