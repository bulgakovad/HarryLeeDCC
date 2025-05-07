from functions import (
    compute_cross_section,
    plot_cross_section_vs_W,
    generate_table,
    compare_strfun,
    compare_exp_model_pdf,
    compare_exp_model_pdf_Bjorken_x,
    compare_exp_model_pdf_Nachtmann_xi,
    compare_exp_pdf_resonance,
    fit_exp_data,
    exp_data_minus_pdf_table
)

def main():
    """
    Main routine: choose an option.
    Available options include:
      'calc', 'plot', 'table',
      'compare_strfun',
      'compare_exp_model_pdf',
      'compare_exp_pdf_resonance'.
    For the comparison functions, you may now enter multiple Q² values separated by commas.
     Available Q2: 2.774,3.244,3.793,4.435,5.187,6.065,7.093,8.294,9.699
    """
    mode = input(
         "Enter 'calc'/'plot'/'table'/'compare_strfun'/'compare_exp_model_pdf'/'compare_exp_pdf_resonance'/'fit_exp_data'/'exp_data_minus_pdf_table' : "
    ).strip().lower()
    
    file_path = "input_data/wempx.dat"
    
    if mode == "calc":
        # Single differential cross section calculation
        try:
            W_input = float(input("Enter W (GeV): "))
            Q2_input = float(input("Enter Q² (GeV²): "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        try:
            xsec = compute_cross_section(
                W_input, Q2_input, beam_energy,
                file_path=file_path, verbose=True
            )
            print("\n=== Results ===")
            print(f"Beam energy: {beam_energy:.5f} GeV")
            print(f"W: {W_input:.5f} GeV, Q²: {Q2_input:.5f} GeV²")
            print(f"dσ/dW/dQ²: {xsec:.5e} (10⁻³⁰ cm²/GeV³)")
        except Exception as e:
            print(f"Error: {e}")

    elif mode == "plot":
        # ANL model cross section vs W
        try:
            Q2_input = float(input("Enter fixed Q² (GeV²) for plotting cross section vs W: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            plot_cross_section_vs_W(Q2_input, beam_energy,
                                    file_path=file_path, num_points=200)
        except Exception as e:
            print(f"Error: {e}")

    elif mode == "table":
        # Generate ANL model table
        try:
            fixed_Q2 = float(input("Enter fixed Q² (GeV²) for generating the table: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            generate_table(file_path, fixed_Q2, beam_energy)
        except Exception as e:
            print(f"Error: {e}")

    elif mode == "compare_strfun":
        # Compare ANL model vs structure-function data
        try:
            q2_input_str = input(
                "Enter fixed Q² values (GeV²) for comparison, separated by commas: "
            ).strip()
            q2_list = [float(s) for s in q2_input_str.split(",") if s]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return

        for q2 in q2_list:
            try:
                print(f"\nProcessing compare_strfun for Q² = {q2} GeV² ...")
                compare_strfun(q2, beam_energy,
                               interp_file=file_path, num_points=200)
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")

    elif mode == "compare_exp_model_pdf":
        # Compare PDF model (and variants) vs experiment
        try:
            xaxis_choice = input(
                "What is on X‑axis? (Enter 'W', 'x', or 'xi'): "
            ).strip().lower()
            q2_input_str = input(
                "Enter fixed Q² values (GeV²) separated by commas: "
            ).strip()
            q2_list = [float(s) for s in q2_input_str.split(",") if s]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return

        for q2 in q2_list:
            try:
                print(f"\nProcessing compare_exp_model_pdf for Q² = {q2} GeV² with x-axis = {xaxis_choice} ...")
                if xaxis_choice == "w":
                    compare_exp_model_pdf(q2, beam_energy, num_points=200)
                elif xaxis_choice == "x":
                    compare_exp_model_pdf_Bjorken_x(q2, beam_energy, num_points=200)
                elif xaxis_choice == "xi":
                    compare_exp_model_pdf_Nachtmann_xi(q2, beam_energy, num_points=200)
                else:
                    print("Invalid X‑axis choice. Please enter 'W', 'x', or 'xi'.")
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")

    elif mode == "compare_exp_pdf_resonance":
        # Compare PDF + resonance vs experiment
        try:
            q2_input_str = input(
                "Enter fixed Q² (GeV²) for resonance comparison (comma-separated): "
            ).strip()
            q2_list = [float(s) for s in q2_input_str.split(",") if s]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return

        for q2 in q2_list:
            try:
                print(f"\nProcessing compare_exp_pdf_resonance for Q² = {q2} GeV² ...")
                compare_exp_pdf_resonance(q2, beam_energy)
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")
    elif mode == "fit_exp_data":
        try:
            q2_input_str = input("Enter Q² values (GeV²) to fit, separated by commas: ").strip()
            q2_list = [float(s) for s in q2_input_str.split(",") if s.strip()]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return

        try:
            print(f"\nRunning fit_exp_data for Q² = {q2_list} GeV² at E = {beam_energy} GeV …")
            fit_exp_data(q2_list, exp_file="exp_data_all.dat", beam_energy=beam_energy)
        except Exception as e:
            print(f"Error during fit_exp_data: {e}")
    elif mode == "exp_data_minus_pdf_table":
        try:
            q2_str = input("Enter Q² values (GeV²), separated by commas: ").strip()
            q2_list = [float(s) for s in q2_str.split(",") if s]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
            output_filename = input("Output filename (default: exp_minus_pdf.txt): ").strip()
            if not output_filename:
                output_filename = "exp_minus_pdf.txt"
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        try:
            exp_data_minus_pdf_table(q2_list, beam_energy, output_filename)
        except Exception as e:
            print(f"Error generating residual table: {e}")
   

    else:
        print("Invalid option. Please enter one of: calc, plot, table, compare_strfun, compare_exp_model_pdf, compare_exp_pdf_resonance.")

if __name__ == "__main__":
    main()
