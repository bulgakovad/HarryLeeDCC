from functions import (
    compute_cross_section,
    plot_cross_section_vs_W,
    generate_table,
    compare_strfun,
    compare_exp_model_pdf,
    compare_exp_model_pdf_Bjorken_x,
    compare_exp_model_pdf_Nachtmann_xi,
    compare_f12_strfun,
    compare_exp_pdf_resonance,
    fit_exp_data_individual,
    fit_exp_data_global,
    anl_struct_func_xsecs,
    plot_xsect_omega_energy
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
     Available experimental Q2: 2.774,3.244,3.793,4.435,5.187,6.065,7.093,8.294,9.699
     Also Q2: 0.5,1,1.5,2,2.5,3
     Also: 1.5,1.75,2.0,2.25,2.5,2.75,3.0
     Also: 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0
    """
    mode = input(
         "Enter 'calc'/'plot'/'table'/'compare_strfun'/'compare_exp_model_pdf'/'compare_f12_strfun'/'compare_exp_pdf_resonance'/'fit_exp_data_individual'/'fit_exp_data_global'/'anl_struct_func_xsecs'/'plot_xsect_omega_energy' : "
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
                    print(f"\nProcessing compare_exp_model_pdf for Q² = {q2_list} GeV² with x-axis = W ...")
                    compare_exp_model_pdf(q2_list, beam_energy, num_points=200)
                elif xaxis_choice == "x":
                    compare_exp_model_pdf_Bjorken_x(q2, beam_energy, num_points=200)
                elif xaxis_choice == "xi":
                    compare_exp_model_pdf_Nachtmann_xi(q2, beam_energy, num_points=200)
                else:
                    print("Invalid X‑axis choice. Please enter 'W', 'x', or 'xi'.")
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")
    elif mode == "compare_f12_strfun":
        try:
            q2_list = [float(s.strip()) for s in input("Enter Q² values, comma-separated: ").split(",")]
        except ValueError:
            print("Invalid input.")
            return
        xaxis_choice = input("What's on the x-axis? Enter 'w' or 'x': ").strip().lower()
        compare_f12_strfun(q2_list, xaxis_choice)
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
                
    elif mode == "fit_exp_data_individual":
        try:
            #q2_input_str = input("Enter Q² values (GeV²) to fit, separated by commas: ").strip()
            #q2_list = [float(s) for s in q2_input_str.split(",") if s.strip()]
            q2_list = [2.774,3.244,3.793,4.435,5.187,6.065,7.093,8.294,9.699]
            #beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
            beam_energy = 10.6
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return
        try:
            print(f"\nRunning fit_exp_data_individual for Q² = {q2_list} GeV² at E = {beam_energy} GeV …")
            fit_exp_data_individual(q2_list, exp_file="bodek_fitting/exp_data_all.dat", beam_energy=beam_energy)
        except Exception as e:
            print(f"Error during fit_exp_data_individual: {e}")
            
    elif mode == "fit_exp_data_global":
        try:
            q2_list = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
            beam_energy = 10.6
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return
        try:
            print(f"\nRunning fit_exp_data_global for Q² = {q2_list} GeV² at E = {beam_energy} GeV …")
            fit_exp_data_global(q2_list, exp_file="bodek_fitting_all_Q2/exp_data_all.dat", beam_energy=beam_energy)
        except Exception as e:
            print(f"Error during fit_exp_data_global: {e}")
            
    elif mode == "anl_struct_func_xsecs":
        q2_list = [float(v) for v in input("Enter Q² (or comma‐separated list): ").split(",")]
        Ebeam   = float(input("Enter beam (lepton) energy (GeV): "))
        for q2 in q2_list:
            print(f"Running anl_struct_func_xsecs for Q²={q2} GeV² …")
            try:
                anl_struct_func_xsecs(q2, Ebeam)
            except Exception as e:
                print(f"   Error for Q²={q2:.3f}: {e}")     
    elif mode == "plot_xsect_omega_energy":
        try:
            q2_input_str = input("Enter Q² values (GeV²), comma-separated: ").strip()
            theta_input_str = input("Enter θ values (deg), comma-separated (same length as Q²): ").strip()
            q2_list = [float(v) for v in q2_input_str.split(",")]
            theta_list = [float(v) for v in theta_input_str.split(",")]
            if len(q2_list) != len(theta_list):
                raise ValueError("Q² list and θ list must be of the same length.")
            E_beam = float(input("Enter beam (lepton) energy (GeV): "))
        except Exception as e:
            print(f"Input error: {e}")
            return
    
        try:
            plot_xsect_omega_energy(q2_list, theta_list, E_beam)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
