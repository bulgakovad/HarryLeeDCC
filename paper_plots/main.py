from harry_lee_functions import (
    compare_strfun,
    compare_exp_model_pdf_bjorken_x,
    compare_exp_model_pdf_nachtmann_xi,
    generate_table_struct_funcs,
    generate_table_xsecs,
)

# 0.5,1.5,2,2.5,2.75,3   for E = 5
# 1.5,2,2.25,2.5,2.774,3  for E=10.6

def main():
    mode = input("Choose option: 'compare_strfun', 'compare_exp_model_pdf_bjorken_x_nachtmann_xi', 'generate_table', or 'generate_xsecs': ").strip().lower()

    if mode in {"compare_strfun", "compare_exp_model_pdf_bjorken_x_nachtmann_xi", "generate_table", "generate_xsecs"}:
        try:
            q2_vals = [
                float(token) for token in
                input("Enter Q² values (GeV²) separated by commas: ").split(",")
                if token.strip()
            ]
        except ValueError:
            print("Please enter only numbers separated by commas.")
            return

        if mode in {"compare_strfun", "compare_exp_model_pdf_bjorken_x_nachtmann_xi", "generate_xsecs"}:
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
            except Exception as err:
                print(f"  Error: {err}")
    else:
        print("Supported options: 'compare_strfun', 'compare_exp_model_pdf_bjorken_x_nachtmann_xi', 'generate_table', 'generate_xsecs'.")


if __name__ == "__main__":
    main()
