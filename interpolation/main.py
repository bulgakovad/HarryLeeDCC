from functions import (
    compute_cross_section,
    plot_cross_section_vs_W,
    generate_table,
    compare_strfun,
    compare_exp_model_pdf,
    compare_exp_model_pdf_Bjorken_x,
    compare_exp_model_pdf_Nachtmann_xi
)

def main():
    """
    Main routine: ask the user whether to calculate a single cross section ("calc"),
    plot cross sections ("plot"), generate a table ("table"), compare structure function data ("compare_strfun"),
    or compare experimental data with the PDF‐based model ("compare_exp_model_pdf").
    
    For the comparison functions, you now can enter multiple Q² values separated by commas.
    In the "compare_exp_model_pdf" branch, you are further prompted to choose what is on the x-axis:
    "W", "x", or "xi".
    Available exp Q2: 2.774,3.244,3.793,4.435,5.187,6.065,7.093,8.294,9.699
    """
    mode = input("Enter 'calc'/'plot'/'table'/'compare_strfun'/'compare_exp_model_pdf': ").strip().lower()
    
    file_path = "input_data/wempx.dat"
    
    if mode == "calc":
        try:
            W_input = float(input("Enter W (GeV): "))
            Q2_input = float(input("Enter Q² (GeV²): "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        try:
            xsec = compute_cross_section(W_input, Q2_input, beam_energy, file_path=file_path, verbose=True)
            print("\n=== Results ===")
            print(f"Beam energy: {beam_energy:.5f} GeV")
            print(f"W: {W_input:.5f} GeV, Q²: {Q2_input:.5f} GeV²")
            print(f"Differential cross section dσ/dW/dQ²: {xsec:.5e} (10^(-30) cm²/GeV³)")
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "plot":
        try:
            Q2_input = float(input("Enter fixed Q² (GeV²) for plotting cross section vs W: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            plot_cross_section_vs_W(Q2_input, beam_energy, file_path=file_path, num_points=200)
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "table":
        try:
            Q2_input = float(input("Enter fixed Q² (GeV²) for generating the table: "))
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
            output_filename = input("Enter output filename (default: table.txt): ").strip()
            if output_filename == "":
                output_filename = "table.txt"
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return
        
        try:
            generate_table(file_path, Q2_input, beam_energy)
        except Exception as e:
            print(f"Error: {e}")
    
    elif mode == "compare_strfun":
        try:
            # Allow multiple Q² values (comma-separated)
            q2_input_str = input("Enter fixed Q² values (GeV²) for comparison, separated by commas: ").strip()
            q2_list = [float(s.strip()) for s in q2_input_str.split(",") if s.strip() != ""]
            beam_energy = float(input("Enter beam (lepton) energy (GeV): "))
        except ValueError:
            print("Invalid input. Please enter numerical values for Q² and beam energy.")
            return
        
        for q2 in q2_list:
            try:
                print(f"\nProcessing compare_strfun for Q² = {q2} GeV² ...")
                compare_strfun(q2, beam_energy, interp_file=file_path, num_points=200)
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")
    
    elif mode == "compare_exp_model_pdf":
        try:
            xaxis_choice = input("What is on X-axis? (Enter 'W', 'x', or 'xi'): ").strip().lower()
            q2_input_str = input("Enter fixed Q² values (GeV²) separated by commas: ").strip()
            q2_list = [float(s.strip()) for s in q2_input_str.split(",") if s.strip() != ""]
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
                    print("Invalid X-axis choice. Please enter 'W', 'x', or 'xi'.")
            except Exception as e:
                print(f"Error for Q² = {q2}: {e}")
    
    else:
        print("Invalid option. Please enter 'calc', 'plot', 'table', 'compare_strfun', or 'compare_exp_model_pdf'.")

if __name__ == "__main__":
    main()
