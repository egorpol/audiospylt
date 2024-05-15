import numpy as np
import pandas as pd
from typing import List, Tuple
from IPython.display import display, HTML
import verovio
import xml.etree.ElementTree as ET

# Constants for pitch and MIDI calculations
A4_FREQUENCY = 440  # Frequency of A4 in Hz
MIDI_BASE = 69      # MIDI value for A4
CENT_FACTOR = 100   # Conversion factor for cents


def process_and_visualize(data: [str, pd.DataFrame], resolution: str = 'half_tone_deviation', include_natural_accidentals: bool = True, save_mei: bool = False, save_path: str = 'output.mei', display_df: bool = True):
    """
    Process the data and visualize it using Verovio.

    Parameters:
        data (str or pd.DataFrame): The data to process, either as a file path or a DataFrame.
        resolution (str): The resolution type for pitch ('half_tone', 'quarter_tone', or 'half_tone_deviation').
        include_natural_accidentals (bool): Flag to include natural accidentals in the MEI notation.
        save_mei (bool): Flag to save the MEI string to a file.
        save_path (str): The path where to save the MEI string.
        display_df (bool): Flag to display the DataFrame in the notebook.
    """
    
    # 1. Load the data or use provided DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data, sep='\t')
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Invalid input data. Provide either a file path (str) or a DataFrame.")

    # 2. Convert frequencies to desired parameters
    pitch_names, midi_values, midi_cent_deviations = convert_frequencies(df)

    # 3. Create an MEI string from the processed data
    notes_and_alters = [pitch_name_and_cent_deviation_to_step_octave_alter(pn, cd, resolution) for pn, cd in zip(pitch_names, midi_cent_deviations)]
    mei_string = create_mei_element(
        notes_and_alters, 
        midi_values, 
        midi_cent_deviations,  # Add this line to pass cent deviations
        'GF',                  # Staff
        resolution,            # Add this line to pass resolution type
        include_natural_accidentals
    )


    # 4. Visualize the data in a table
    frequencies = df['Frequency (Hz)'].tolist()
    df_para = pd.DataFrame({
        'Frequency (Hz)': frequencies,
        'MIDI Value': midi_values,
        'MIDI Cent Deviation': midi_cent_deviations,
        'Pitch Name': pitch_names,
        'Note': [x[0] for x in notes_and_alters],
        'Octave': [x[1] for x in notes_and_alters],
        'Alter': [x[2] for x in notes_and_alters],
    })
    if display_df:
        display(df_para)
        
    # 5. Render the data using Verovio
    vrvToolkit = verovio.toolkit()
    options = {
        "inputFormat": "mei",
        "scale": 50,
        "adjustPageHeight": True,
        "noHeader": True,
        "border": 0,
        "pageHeight": 10000,
        "pageWidth": 1800,
        "staffLineWidth": 1.5,
        "systemDividerLineWidth": 1.5
    }
    htmlCode = vrvToolkit.renderData(mei_string, options=options)
    display(HTML(htmlCode))

    # 6. Provide an option to save the MEI string to a file
    if save_mei:
        with open(save_path, 'w') as f:
            f.write(mei)


# Helper Functions

def acoustical_pitch_name(midi_num: int) -> str:
    """
    Converts a MIDI number to its acoustical pitch name.
    
    Parameters:
        midi_num (int): MIDI note number.
    
    Returns:
        str: Corresponding pitch name.
    """
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_idx = midi_num % 12
    octave = (midi_num - 12) // 12
    return f"{pitch_names[pitch_idx]}{octave}"
    
def midi_value(log_freq_ratio: float) -> int:
    """
    Converts a logarithmic frequency ratio to a MIDI value.

    The MIDI value is calculated using the formula:
    MIDI = 12 * log2(freq / A4_FREQ) + 69,
    where freq is the frequency in Hz and A4_FREQ is the reference A4 frequency (440 Hz).

    Parameters:
        log_freq_ratio (float): The logarithmic frequency ratio (log2(freq / A4_FREQ)).

    Returns:
        int: The calculated MIDI value, rounded to the nearest integer.
    """
    return int(round(12 * log_freq_ratio + 69))

def midi_cent_value(log_freq_ratio: float) -> int:
    """
    Converts a logarithmic frequency ratio to a MIDI cent value.

    The MIDI cent value is calculated by first converting the frequency ratio to a MIDI value,
    and then scaling it up by a factor of 100 (as there are 100 cents in a semitone).

    Parameters:
        log_freq_ratio (float): The logarithmic frequency ratio (log2(freq / A4_FREQ)).

    Returns:
        int: The calculated MIDI cent value, rounded to the nearest integer.
    """
    return int(round(69 * 100 + 1200 * log_freq_ratio))

def convert_frequencies(df: pd.DataFrame) -> Tuple[List[str], List[int], List[str]]:
    """
    Converts frequencies from a DataFrame to pitch names, MIDI values, and cent deviations.

    This function performs the following steps:
    1. Extract frequencies from the DataFrame.
    2. Calculate the logarithmic frequency ratios for each frequency.
    3. Convert these ratios to MIDI values and MIDI cent values.
    4. Obtain pitch names for each MIDI value.
    5. Calculate the cent deviations for each note, which represent the difference
       from the standard tempered scale in cents.

    Parameters:
        df (pd.DataFrame): DataFrame containing frequencies under 'Frequency (Hz)' column.

    Returns:
        Tuple[List[str], List[int], List[str]]: A tuple containing lists of pitch names,
                                                MIDI values, and formatted cent deviations.
    """
    frequencies = df['Frequency (Hz)'].tolist()
    log_freq_ratios = [np.log2(freq / A4_FREQUENCY) for freq in frequencies]
    midi_vals = [midi_value(lfr) for lfr in log_freq_ratios]
    midi_cent_vals = [midi_cent_value(lfr) for lfr in log_freq_ratios]
    pitch_names = [acoustical_pitch_name(mv) for mv in midi_vals]
    midi_cent_deviations = [((mcv - mv * 100) + 50) % 100 - 50 for mv, mcv in zip(midi_vals, midi_cent_vals)]
    formatted_deviations = [f'+{x}' if x >= 0 else f'{x}' for x in midi_cent_deviations]
    return pitch_names, midi_vals, formatted_deviations



def pitch_name_and_cent_deviation_to_step_octave_alter(pitch_name: str, cent_deviation: str, resolution: str = 'quarter_tone') -> Tuple[str, int, str]:
    """
    Converts pitch name and cent deviation to musical step, octave, and alteration.

    Parameters:
        pitch_name (str): The pitch name (e.g., C#, A, G).
        cent_deviation (str): The cent deviation as a string.
        resolution (str): The resolution type, 'quarter_tone' or another specified type.

    Returns:
        Tuple[str, int, str]: A tuple containing the musical step, octave, and alteration.
    """
    # Extract the step (note) and octave from the pitch name
    step = pitch_name[0]
    octave = int(pitch_name[-1])
    deviation = int(cent_deviation)
    alter = 'n'  # Default alteration is natural

    # Determine the alteration based on the pitch name and deviation
    # Sharp and flat alterations
    
    if '#' in pitch_name:
        alter = 's'  # Sharp
    elif 'b' in pitch_name:
        alter = 'f'  # Flat

    # Quarter tone resolution adjustments
    if resolution == 'quarter_tone':
        if '#' in pitch_name:
            if deviation >= 25: alter = '3qs'
            elif deviation > 0: alter = '1qs'
            else: alter = 's'
        elif 'b' in pitch_name:
            if deviation <= -25: alter = '3qf'
            elif deviation < 0: alter = '1qf'
            else: alter = 'f'
        else:
            if deviation >= 25 or deviation > 0: alter = '1qs'
            elif deviation <= -25 or deviation < 0: alter = '1qf'

    return step, octave, alter

def map_alteration_to_mei_accid(alter: str, include_natural_accidentals: bool = False) -> str:
    """
    Maps a musical alteration to its corresponding MEI accidental representation.

    Parameters:
        alter (str): The musical alteration (e.g., sharp, flat).
        include_natural_accidentals (bool): Flag to include natural accidentals in the mapping.

    Returns:
        str: The MEI accidental representation.
    """
    mapping = {
        's': "s",     # sharp
        'f': "f",     # flat
        '1qs': "1qs", # 1/4-tone sharp accidental
        '1qf': "1qf", # 1/4-tone flat accidental
        '3qs': "3qs", # 3/4-tone sharp accidental
        '3qf': "3qf", # 3/4-tone flat accidental
        'nu': "nu",   # Natural note raised by quarter tone
        'nd': "nd",   # Natural note lowered by quarter tone
        'su': "su",   # Sharp note raised by quarter tone
        'sd': "sd",   # Sharp note lowered by quarter tone
        # ... other mappings ...
    }

    # Handling the natural accidental based on the flag
    if include_natural_accidentals:
        mapping['n'] = "n"  # Include natural accidentals
    else:
        mapping['n'] = ""   # No natural accidentals

    return mapping.get(alter, "")


def create_mei_element(notes_and_alters, midi_values, cent_deviations, staff, resolution_type, include_natural_accidentals):
    """
    Creates an MEI (Music Encoding Initiative) element for music notation.

    Parameters:
        notes_and_alters (list of tuples): List containing tuples of (step, octave, alter).
        midi_values (list of int): List of MIDI values.
        cent_deviations (list of int): List of cent deviations.
        staff (str): The staff type (e.g., 'G', 'F').
        resolution_type (str): The resolution type for the notation.
        include_natural_accidentals (bool): Flag to include natural accidentals.

    Returns:
        str: An MEI string representing the music notation.
    """
    # Create the MEI root element and its child elements
    mei = ET.Element('mei', attrib={'xmlns': "http://www.music-encoding.org/ns/mei", 'meiversion': "5.0"})
    music = ET.SubElement(mei, 'music')
    body = ET.SubElement(music, 'body')
    mdiv = ET.SubElement(body, 'mdiv')
    score = ET.SubElement(mdiv, 'score')
    scoreDef = ET.SubElement(score, 'scoreDef', attrib={'keysig': "0", 'mode': "major"})
    staffGrp = ET.SubElement(scoreDef, 'staffGrp', attrib={'n': "1"})

    # Adding staff definitions
    if 'G' in staff:
        ET.SubElement(staffGrp, 'staffDef', attrib={'n': "1", 'lines': "5", 'clef.shape': "G", 'clef.line': "2"})
    if 'F' in staff:
        ET.SubElement(staffGrp, 'staffDef', attrib={'n': "2", 'lines': "5", 'clef.shape': "F", 'clef.line': "4"})

    section = ET.SubElement(score, 'section')

    # Loop through notes and alterations to create MEI elements for each note
    for i, ((step, octave, alter), midi_value, cent_deviation) in enumerate(zip(notes_and_alters, midi_values, cent_deviations)):
    
    # Logic for creating MEI elements based on the note, octave, alteration, and staff
        measure = ET.SubElement(section, 'measure', attrib={'n': str(i+1), 'right': "invis"})
        accid_value = map_alteration_to_mei_accid(alter, include_natural_accidentals)

        if 'G' in staff:
            staff_elem_g = ET.SubElement(measure, 'staff', attrib={'n': "1"})
            layer_g = ET.SubElement(staff_elem_g, 'layer', attrib={'n': "1"})
            if 'F' not in staff or octave >= 4 or (octave == 3 and step == 'B'):
                # Add note to G staff
                note_elem = ET.SubElement(layer_g, 'note', attrib={
                    'dur': "4", 'oct': str(octave), 'pname': step.lower(), 'pnum': str(midi_value),
                    'stem.dir': "down", 'accid': accid_value, 'stem.visible': "false"
                })
    
        if 'F' in staff:
            staff_elem_f = ET.SubElement(measure, 'staff', attrib={'n': "2"})
            layer_f = ET.SubElement(staff_elem_f, 'layer', attrib={'n': "1"})
            if 'G' not in staff or octave < 4 or (octave == 4 and step == 'C' and alter in ['f', '1qf', '3qf']):
                # Add note to F staff
                note_elem = ET.SubElement(layer_f, 'note', attrib={
                    'dur': "4", 'oct': str(octave), 'pname': step.lower(), 'pnum': str(midi_value),
                    'stem.dir': "down", 'accid': accid_value, 'stem.visible': "false"
                })
                    
        if resolution_type == 'half_tone_deviation':
            note_id = f"note_{i}"
            note_elem.set('xml:id', note_id)
            fing = ET.SubElement(measure, 'fing', attrib={
                'startid': f"#{note_id}",
                'place': "above"
            })
            fing.text = str(cent_deviation)

    return ET.tostring(mei, encoding='unicode')


