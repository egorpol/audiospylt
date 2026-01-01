import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from IPython.display import display, HTML
import verovio
import xml.etree.ElementTree as ET
import logging

# Setup basic logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# For more detailed debug logs during development, uncomment the next line
# logging.getLogger().setLevel(logging.DEBUG)

# Constants
A4_FREQUENCY = 440
MIDI_BASE = 69
CENT_FACTOR = 100
XML_NS = "http://www.w3.org/XML/1998/namespace" # XML namespace constant
ET.register_namespace('xml', XML_NS) # Register globally

# --- Pitch Utility Functions ---

def acoustical_pitch_name(midi_num: int) -> str:
    """Converts MIDI number to acoustical pitch name (e.g., C#4)."""
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_idx = midi_num % 12
    octave = (midi_num - 12) // 12
    return f"{pitch_names[pitch_idx]}{octave}"

def midi_value(log_freq_ratio: float) -> int:
    """Converts log frequency ratio relative to A4 to MIDI value."""
    return int(round(12 * log_freq_ratio + MIDI_BASE))

def midi_cent_value(log_freq_ratio: float) -> int:
    """Converts log frequency ratio relative to A4 to MIDI cent value."""
    return int(round(MIDI_BASE * CENT_FACTOR + 12 * CENT_FACTOR * log_freq_ratio))

def convert_frequencies(df: pd.DataFrame) -> Tuple[List[str], List[Optional[int]], List[Optional[int]]]:
    """
    Converts frequencies from a DataFrame to pitch names, MIDI values, and raw cent deviations.
    Handles non-positive frequencies by returning None for corresponding values.
    """
    if 'Frequency (Hz)' not in df.columns:
        raise ValueError("DataFrame must contain a 'Frequency (Hz)' column.")

    frequencies = df['Frequency (Hz)'].tolist()
    pitch_names_list: List[str] = []
    midi_vals_list: List[Optional[int]] = []
    midi_cent_deviations_list: List[Optional[int]] = []
    
    log2_A4 = np.log2(A4_FREQUENCY)

    for i, freq in enumerate(frequencies):
        if freq is None or pd.isna(freq) or not isinstance(freq, (int, float)) or freq <= 0:
            logging.warning(f"Invalid frequency at index {i}: {freq}. Assigning N/A values.")
            pitch_names_list.append("N/A")
            midi_vals_list.append(None)
            midi_cent_deviations_list.append(None)
            continue

        try:
            log_freq_ratio = np.log2(freq) - log2_A4
            mv = midi_value(log_freq_ratio)
            mcv = midi_cent_value(log_freq_ratio)
            pn = acoustical_pitch_name(mv)
            deviation = int(round( ((mcv - mv * CENT_FACTOR) + 50) % 100 - 50 ))
            
            pitch_names_list.append(pn)
            midi_vals_list.append(mv)
            midi_cent_deviations_list.append(deviation)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            logging.error(f"Error processing frequency {freq} at index {i}: {e}")
            pitch_names_list.append("N/A")
            midi_vals_list.append(None)
            midi_cent_deviations_list.append(None)

    return pitch_names_list, midi_vals_list, midi_cent_deviations_list


def pitch_name_and_deviation_to_step_octave_alter(
    pitch_name: str,
    cent_deviation: Optional[int],
    resolution: str = 'quarter_tone'
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Converts pitch name and cent deviation to MEI step, octave, and an MEI-compatible
    accidental string. Supports 'half_tone', 'quarter_tone', and 'eighth_tone' resolutions.
    """
    if pitch_name == "N/A" or cent_deviation is None:
        return None, None, None

    step = pitch_name[0]
    octave_str = ''.join(filter(str.isdigit, pitch_name))
    try:
        octave = int(octave_str)
    except ValueError:
        logging.error(f"Could not parse octave from pitch name: {pitch_name}")
        return None, None, None

    base_alter_from_name = 'n'
    if '#' in pitch_name: base_alter_from_name = 's'
    elif 'b' in pitch_name: base_alter_from_name = 'f'
    
    alter = base_alter_from_name # Default MEI "n" or from pitch name
    deviation = cent_deviation

    QT_THRESHOLD = 25
    ET_THRESHOLD = 12 # Approx 1/8th tone

    if resolution == 'quarter_tone':
        if base_alter_from_name == 's':
            if deviation >= QT_THRESHOLD: alter = '3qs'
            elif deviation > 0: alter = '1qs' # Sharp but not enough for 3qs remains 1qs (closer to sharp)
            # If deviation is negative, it means it's flatter than sharp, but still sharp context
        elif base_alter_from_name == 'f':
            if deviation <= -QT_THRESHOLD: alter = '3qf'
            elif deviation < 0: alter = '1qf' # Flat but not enough for 3qf remains 1qf (closer to flat)
        else: # base is 'n'
            if deviation >= QT_THRESHOLD: alter = '1qs' # Natural raised by >= quarter tone
            elif deviation > 0: alter = '1qs'          # Natural raised by < quarter tone (still 1qs as per original)
            elif deviation <= -QT_THRESHOLD: alter = '1qf'# Natural lowered by >= quarter tone
            elif deviation < 0: alter = '1qf'           # Natural lowered by < quarter tone (still 1qf as per original)

    elif resolution == 'eighth_tone':
        if base_alter_from_name == 'n':
            if deviation >= QT_THRESHOLD: alter = '1qs'
            elif deviation >= ET_THRESHOLD: alter = 'nu'
            elif deviation <= -QT_THRESHOLD: alter = '1qf'
            elif deviation <= -ET_THRESHOLD: alter = 'nd'
        elif base_alter_from_name == 's':
            if deviation >= QT_THRESHOLD: alter = '3qs'
            elif deviation >= ET_THRESHOLD: alter = 'su'
            elif deviation <= -QT_THRESHOLD: alter = 'n' # Simplified: sharp lowered significantly -> natural
            elif deviation <= -ET_THRESHOLD: alter = 'sd'
        elif base_alter_from_name == 'f':
            if deviation <= -QT_THRESHOLD: alter = '3qf'
            elif deviation < -ET_THRESHOLD: alter = 'fd' # Original: deviation < -ET_THRESHOLD
            elif deviation >= QT_THRESHOLD: alter = 'n'  # Simplified: flat raised significantly -> natural
            elif deviation >= ET_THRESHOLD: alter = 'fu'

    logging.debug(f"Pitch: {pitch_name}, Dev: {cent_deviation}, Res: {resolution} -> Step: {step}, Oct: {octave}, MEI Alter: {alter}")
    return step, octave, alter

def map_alteration_to_mei_accid(alter: Optional[str], include_natural_accidentals: bool = False) -> str:
    """Maps the 'alter' string to an MEI 'accid' attribute value."""
    if alter is None or alter == "": return ""
    if alter == 'n' and not include_natural_accidentals: return ""
    return alter

# --- MEI Structure Helper Functions ---

def _init_mei_structure_elements() -> Tuple[ET.Element, ET.Element]:
    """Initializes base MEI document structure."""
    mei = ET.Element('mei', attrib={'xmlns': "http://www.music-encoding.org/ns/mei", 'meiversion': "5.0"})
    music = ET.SubElement(mei, 'music')
    body = ET.SubElement(music, 'body')
    mdiv = ET.SubElement(body, 'mdiv')
    score = ET.SubElement(mdiv, 'score')
    return mei, score

def _create_staff_defs_and_map(clef_config: str) -> Tuple[List[Dict], Dict[str, int], Optional[int], Optional[int]]:
    """Creates staffDef attributes and maps clef type to staff number."""
    staff_defs_list: List[Dict] = []
    staff_n_map: Dict[str, int] = {}
    g_staff_n: Optional[int] = None
    f_staff_n: Optional[int] = None
    current_staff_idx = 1

    if 'G' in clef_config:
        g_staff_n = current_staff_idx
        staff_defs_list.append({'n': str(g_staff_n), 'lines': "5", 'clef.shape': "G", 'clef.line': "2"})
        staff_n_map['G'] = g_staff_n
        current_staff_idx += 1
    
    if 'F' in clef_config:
        f_staff_n = current_staff_idx
        staff_defs_list.append({'n': str(f_staff_n), 'lines': "5", 'clef.shape': "F", 'clef.line': "4"})
        staff_n_map['F'] = f_staff_n
        # current_staff_idx += 1 # Not strictly needed if F is the last possible clef

    if not staff_defs_list:
        raise ValueError("Invalid clef_config. Must include 'G' or 'F'.")
    return staff_defs_list, staff_n_map, g_staff_n, f_staff_n

def _add_score_def_to_score(score_element: ET.Element, clef_config: str) -> Tuple[List[Dict], Dict[str, int], Optional[int], Optional[int]]:
    """Adds scoreDef and staffGrp to the score element."""
    scoreDef = ET.SubElement(score_element, 'scoreDef', attrib={'keysig': "0", 'mode': "major"})
    staff_defs_list, staff_n_map, g_staff_n, f_staff_n = _create_staff_defs_and_map(clef_config)
    
    staffGrp_attrs = {'n': str(len(staff_defs_list))}
    if len(staff_defs_list) > 1: # e.g., for 'GF'
        staffGrp_attrs['symbol'] = "brace"
    
    staffGrp = ET.SubElement(scoreDef, 'staffGrp', attrib=staffGrp_attrs)
    for staff_def_attrs in staff_defs_list:
        ET.SubElement(staffGrp, 'staffDef', attrib=staff_def_attrs)
    return staff_defs_list, staff_n_map, g_staff_n, f_staff_n


# --- MEI Content Helper Functions (Chord and Sequence specific) ---

def _add_chord_content_to_measure(
    measure_el: ET.Element,
    notes_data_group: List[Tuple[Optional[str], Optional[int], Optional[str]]],
    midi_values_group: List[Optional[int]],
    cent_deviations_group: List[Optional[int]],
    clef_config: str,
    g_staff_n: Optional[int], # Direct staff numbers
    f_staff_n: Optional[int],
    include_natural_accidentals: bool,
    resolution_type: str,
    display_cent_deviation_for_chord: bool,
    current_note_counter: int
) -> int:
    """Populates a measure with chord(s) and optional deviation fingerings."""
    layer_g_chord, layer_f_chord = None, None
    staff_elements: Dict[int, ET.Element] = {} # To store created staff elements

    # Create staff and layer elements as needed
    if clef_config == 'GF':
        if g_staff_n: 
            staff_g = ET.SubElement(measure_el, 'staff', {'n': str(g_staff_n)}); staff_elements[g_staff_n] = staff_g
            layer_g_chord = ET.SubElement(staff_g, 'layer', {'n':"1"})
        if f_staff_n: 
            staff_f = ET.SubElement(measure_el, 'staff', {'n': str(f_staff_n)}); staff_elements[f_staff_n] = staff_f
            layer_f_chord = ET.SubElement(staff_f, 'layer', {'n':"1"})
    elif clef_config == 'G' and g_staff_n:
        staff_g = ET.SubElement(measure_el, 'staff', {'n': str(g_staff_n)}); staff_elements[g_staff_n] = staff_g
        layer_g_chord = ET.SubElement(staff_g, 'layer', {'n':"1"})
    elif clef_config == 'F' and f_staff_n:
        staff_f = ET.SubElement(measure_el, 'staff', {'n': str(f_staff_n)}); staff_elements[f_staff_n] = staff_f
        layer_f_chord = ET.SubElement(staff_f, 'layer', {'n':"1"})

    chord_notes_g_packaged = [] # List of (note_dict, note_id_str, fing_info_or_None)
    chord_notes_f_packaged = []
    all_deviation_fingerings_for_measure = []

    for note_info, midi_val, cent_dev in zip(notes_data_group, midi_values_group, cent_deviations_group):
        step, octave, alter = note_info
        if step is None or octave is None or midi_val is None:
            logging.warning(f"Skipping invalid note data in chord generation: {note_info}, MIDI: {midi_val}")
            continue

        target_staff_n_for_note = None
        if clef_config == 'GF': target_staff_n_for_note = g_staff_n if midi_val > 60 else f_staff_n
        elif clef_config == 'G': target_staff_n_for_note = g_staff_n
        elif clef_config == 'F': target_staff_n_for_note = f_staff_n
        
        if target_staff_n_for_note is None:
            logging.warning(f"Could not determine target staff for chord note (MIDI:{midi_val}).")
            continue

        accid_value = map_alteration_to_mei_accid(alter, include_natural_accidentals)
        note_id_str = f"note_{current_note_counter}"
        current_note_counter += 1

        note_dict = {'oct': str(octave), 'pname': step.lower(), 'pnum': str(midi_val)}
        if accid_value: note_dict['accid'] = accid_value
        
        deviation_fing_info = None
        if display_cent_deviation_for_chord and resolution_type == 'half_tone_deviation' and cent_dev is not None:
            place_hint = "above"
            if target_staff_n_for_note == f_staff_n and midi_val < 48: place_hint = "below"
            elif target_staff_n_for_note == g_staff_n and midi_val >= 84: place_hint = "below"
            deviation_fing_info = {
                'text': f"+{cent_dev}" if cent_dev >= 0 else str(cent_dev),
                'staff': str(target_staff_n_for_note), 'layer': "1",
                'startid': f"#{note_id_str}", 'place': place_hint
            }
        
        note_package = (note_dict, note_id_str, deviation_fing_info)
        if target_staff_n_for_note == g_staff_n and layer_g_chord is not None:
            chord_notes_g_packaged.append(note_package)
        elif target_staff_n_for_note == f_staff_n and layer_f_chord is not None:
            chord_notes_f_packaged.append(note_package)

    id_key = f"{{{XML_NS}}}id"
    chord_attribs = {'dur':"4", 'stem.visible':'false'}

    if layer_g_chord is not None and chord_notes_g_packaged:
        chord_g_el = ET.SubElement(layer_g_chord, 'chord', attrib=chord_attribs)
        for nd_dict, nd_id, nd_fing_info in chord_notes_g_packaged:
            ET.SubElement(chord_g_el, 'note', attrib=nd_dict).set(id_key, nd_id)
            if nd_fing_info: all_deviation_fingerings_for_measure.append(nd_fing_info)
    
    if layer_f_chord is not None and chord_notes_f_packaged:
        chord_f_el = ET.SubElement(layer_f_chord, 'chord', attrib=chord_attribs)
        for nd_dict, nd_id, nd_fing_info in chord_notes_f_packaged:
            ET.SubElement(chord_f_el, 'note', attrib=nd_dict).set(id_key, nd_id)
            if nd_fing_info: all_deviation_fingerings_for_measure.append(nd_fing_info)

    if all_deviation_fingerings_for_measure: # These are only populated if display_cent_deviation_for_chord was true
        for fing_attrs in all_deviation_fingerings_for_measure:
            ET.SubElement(measure_el, 'fing', attrib=fing_attrs)
            logging.debug(f"Chord Mode: Added fingering '{fing_attrs['text']}' ref {fing_attrs['startid']}")
            
    return current_note_counter


def _add_sequence_content_to_section(
    section_el: ET.Element,
    notes_data_seq: List[Tuple[Optional[str], Optional[int], Optional[str]]],
    midi_values_seq: List[Optional[int]],
    cent_deviations_seq: List[Optional[int]],
    clef_config: str,
    g_staff_n: Optional[int],
    f_staff_n: Optional[int],
    include_natural_accidentals: bool,
    resolution_type: str,
    display_cent_deviation_for_sequence: bool,
    current_note_counter: int
) -> int:
    """Populates a section with a sequence of notes, each in its own measure."""
    id_key = f"{{{XML_NS}}}id"
    for i, (note_info, midi_val, cent_dev) in enumerate(zip(notes_data_seq, midi_values_seq, cent_deviations_seq)):
        measure_n_str = str(i + 1)
        measure_el = ET.SubElement(section_el, 'measure', attrib={'n': measure_n_str, 'right': "invis"})
        
        step, octave, alter = note_info
        staff_el_g, staff_el_f, layer_el_g, layer_el_f = None, None, None, None

        # Always create staff structures for GF clef to ensure measure consistency
        if clef_config == 'GF':
            if g_staff_n: staff_el_g = ET.SubElement(measure_el, 'staff', {'n': str(g_staff_n)}); layer_el_g = ET.SubElement(staff_el_g, 'layer', {'n':"1"})
            if f_staff_n: staff_el_f = ET.SubElement(measure_el, 'staff', {'n': str(f_staff_n)}); layer_el_f = ET.SubElement(staff_el_f, 'layer', {'n':"1"})
        elif clef_config == 'G' and g_staff_n:
            staff_el_g = ET.SubElement(measure_el, 'staff', {'n': str(g_staff_n)}); layer_el_g = ET.SubElement(staff_el_g, 'layer', {'n':"1"})
        elif clef_config == 'F' and f_staff_n:
            staff_el_f = ET.SubElement(measure_el, 'staff', {'n': str(f_staff_n)}); layer_el_f = ET.SubElement(staff_el_f, 'layer', {'n':"1"})

        if step is None or octave is None or midi_val is None:
            logging.warning(f"Skipping MEI note for sequence item {i+1} in measure {measure_n_str} due to invalid pitch data.")
            continue # Staffs/layers are already created if needed for this empty measure.

        target_staff_n_for_note, target_layer_el = None, None
        if clef_config == 'GF': 
            target_staff_n_for_note = g_staff_n if midi_val > 60 else f_staff_n
            target_layer_el = layer_el_g if midi_val > 60 else layer_el_f
        elif clef_config == 'G': 
            target_staff_n_for_note = g_staff_n
            target_layer_el = layer_el_g
        elif clef_config == 'F': 
            target_staff_n_for_note = f_staff_n
            target_layer_el = layer_el_f
        
        if target_staff_n_for_note is None or target_layer_el is None:
            logging.warning(f"Could not assign sequence note (MIDI:{midi_val}) to staff/layer in measure {measure_n_str}.")
            continue
            
        accid_value = map_alteration_to_mei_accid(alter, include_natural_accidentals)
        note_id_str = f"note_{current_note_counter}"
        current_note_counter += 1

        note_attrs = {'dur': "4", 'oct': str(octave), 'pname': step.lower(), 'pnum': str(midi_val), 'stem.visible': "false"}
        if accid_value: note_attrs['accid'] = accid_value
        
        note_el = ET.SubElement(target_layer_el, 'note', attrib=note_attrs)
        note_el.set(id_key, note_id_str)

        if display_cent_deviation_for_sequence and resolution_type == 'half_tone_deviation' and cent_dev is not None:
            place_hint = "above" # Simplified for sequence
            fing_attrs = {'staff': str(target_staff_n_for_note), 'layer': "1", 'startid': f"#{note_id_str}", 'place': place_hint}
            fing_el = ET.SubElement(measure_el, 'fing', attrib=fing_attrs)
            fing_el.text = f"+{cent_dev}" if cent_dev >= 0 else str(cent_dev)
            logging.debug(f"Seq Mode: Added fingering '{fing_el.text}' ref {note_id_str} to measure {measure_n_str}")
            
    return current_note_counter

# --- Main MEI Creation Functions (Part 1 and Part 2) ---

def create_mei_string(
    notes_data: List[Tuple[Optional[str], Optional[int], Optional[str]]],
    midi_values: List[Optional[int]],
    cent_deviations: List[Optional[int]],
    clef_config: str,
    resolution_type: str,
    include_natural_accidentals: bool,
    display_cent_deviation: bool, # Effective flag from caller
    display_mode: str
) -> str:
    """Creates an MEI string for sequence or chord display."""
    mei_root, score_el = _init_mei_structure_elements()
    _, _, g_staff_n, f_staff_n = _add_score_def_to_score(score_el, clef_config) # staff_n_map not directly needed here
    section = ET.SubElement(score_el, 'section')
    note_counter = 0

    if display_mode == 'sequence':
        logging.debug("Creating MEI for sequence mode")
        _add_sequence_content_to_section(
            section_el=section,
            notes_data_seq=notes_data,
            midi_values_seq=midi_values,
            cent_deviations_seq=cent_deviations,
            clef_config=clef_config,
            g_staff_n=g_staff_n,
            f_staff_n=f_staff_n,
            include_natural_accidentals=include_natural_accidentals,
            resolution_type=resolution_type,
            display_cent_deviation_for_sequence=display_cent_deviation, # Use the effective flag
            current_note_counter=note_counter
        )
    elif display_mode == 'chord':
        logging.debug("Creating MEI for chord mode")
        measure = ET.SubElement(section, 'measure', attrib={'n': "1", 'right':"invis"})
        _add_chord_content_to_measure(
            measure_el=measure,
            notes_data_group=notes_data,
            midi_values_group=midi_values,
            cent_deviations_group=cent_deviations,
            clef_config=clef_config,
            g_staff_n=g_staff_n,
            f_staff_n=f_staff_n,
            include_natural_accidentals=include_natural_accidentals,
            resolution_type=resolution_type,
            display_cent_deviation_for_chord=display_cent_deviation, # Use the effective flag (usually False for chords)
            current_note_counter=note_counter
        )
    
    mei_string = ET.tostring(mei_root, encoding='unicode', method='xml')
    logging.debug(f"--- Generated MEI Snippet ({display_mode}) ---")
    logging.debug(mei_string[:1000] + ("..." if len(mei_string) > 1000 else ""))
    logging.debug(f"--- End MEI Snippet ({display_mode}) ---")
    return mei_string

def create_temporal_mei_string(
    grouped_notes_data: List[Tuple[Any, List[Tuple[Optional[str], Optional[int], Optional[str]]], List[Optional[int]], List[Optional[int]]]],
    clef_config: str,
    resolution_type: str, # Renamed from 'resolution' in the call
    include_natural_accidentals: bool,
    measure_represents_sec: float # <-- NEW PARAMETER
) -> str:
    """Creates an MEI string for temporally grouped chords with padding."""
    mei_root, score_el = _init_mei_structure_elements()
    _, _, g_staff_n, f_staff_n = _add_score_def_to_score(score_el, clef_config)
    section = ET.SubElement(score_el, 'section')
    note_counter = 0
    measure_counter = 0

    for group_idx, (time_key, notes_data_group, midi_values_group, cent_deviations_group) in enumerate(grouped_notes_data):
        time_start, time_stop = time_key
        actual_duration_sec = time_stop - time_start

        if actual_duration_sec <= 0:
            logging.warning(f"Skipping time group {group_idx+1} due to zero/negative duration: {actual_duration_sec:.3f}s")
            continue

        measure_counter += 1
        measure_chord_el = ET.SubElement(section, 'measure', attrib={'n': str(measure_counter), 'right': "invis"})
        
        note_counter = _add_chord_content_to_measure(
            measure_el=measure_chord_el,
            notes_data_group=notes_data_group,
            midi_values_group=midi_values_group,
            cent_deviations_group=cent_deviations_group,
            clef_config=clef_config,
            g_staff_n=g_staff_n,
            f_staff_n=f_staff_n,
            include_natural_accidentals=include_natural_accidentals,
            resolution_type=resolution_type,
            display_cent_deviation_for_chord=False, 
            current_note_counter=note_counter
        )

        # Use the passed parameter here instead of the global constant
        num_padding_measures = int(round(actual_duration_sec / measure_represents_sec)) - 1
        if num_padding_measures < 0: num_padding_measures = 0
        logging.debug(f"Duration {actual_duration_sec:.3f}s needs {num_padding_measures} padding measures (unit: {measure_represents_sec}s).")


        for _ in range(num_padding_measures):
            measure_counter += 1
            pad_measure = ET.SubElement(section, 'measure', attrib={'n': str(measure_counter), 'right': "invis"})
            if clef_config == 'GF':
                if g_staff_n: ET.SubElement(ET.SubElement(pad_measure, 'staff', {'n': str(g_staff_n)}), 'layer', {'n':"1"})
                if f_staff_n: ET.SubElement(ET.SubElement(pad_measure, 'staff', {'n': str(f_staff_n)}), 'layer', {'n':"1"})
            elif clef_config == 'G' and g_staff_n:
                ET.SubElement(ET.SubElement(pad_measure, 'staff', {'n': str(g_staff_n)}), 'layer', {'n':"1"})
            elif clef_config == 'F' and f_staff_n:
                ET.SubElement(ET.SubElement(pad_measure, 'staff', {'n': str(f_staff_n)}), 'layer', {'n':"1"})

    mei_string = ET.tostring(mei_root, encoding='unicode', method='xml')
    logging.debug("--- Generated Temporal MEI Snippet (Quantized Duration) ---")
    logging.debug(mei_string[:1000] + ("..." if len(mei_string) > 1000 else ""))
    logging.debug("--- End Temporal MEI Snippet (Quantized Duration) ---")
    return mei_string

# --- Data Loading and Rendering/Saving Utilities ---

def _load_dataframe_from_input(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Loads a DataFrame from a file path or uses the provided DataFrame."""
    df: pd.DataFrame
    if isinstance(data, str):
        sep: str = '\t'
        try:
            with open(data, 'r', encoding='utf-8') as fh:
                first_line: str = fh.readline()
                if first_line and ',' in first_line and ('\t' not in first_line or first_line.count(',') > first_line.count('\t')): # Heuristic for CSV
                    sep = ','
            df = pd.read_csv(data, sep=sep)
            logging.info(f"Loaded data from file '{data}' (delimiter: {repr(sep)})")
        except FileNotFoundError:
            logging.error(f"File not found: {data}")
            raise
        except Exception as e:
            logging.error(f"Error reading file '{data}': {e}")
            raise
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        logging.info("Using provided DataFrame.")
    else:
        raise TypeError("Invalid input data type. Must be a file path (str) or pandas.DataFrame.")
    return df

def _render_and_save_mei(
    mei_string: str,
    verovio_options: Dict,
    save_mei: bool,
    mei_save_path: str, # Renamed from save_path for clarity
    save_svg: bool = False, # New parameter
    svg_save_path: str = 'output.svg' # New parameter
    # Add save_png and png_save_path if doing direct PNG conversion here
):
    """
    Renders MEI using Verovio, optionally saves the MEI string,
    and optionally saves the rendered SVG.
    """
    if not mei_string:
        logging.info("MEI string is empty. Skipping Verovio rendering and saving.")
        return None # Return None if no SVG is generated

    logging.info("Rendering MEI with Verovio...")
    vrvToolkit = verovio.toolkit()
    logging.debug(f"Verovio Options: {verovio_options}")
    svg_code = "" # Initialize svg_code

    try:
        ET.fromstring(mei_string) # Basic XML validation
        logging.debug("MEI XML validation successful.")
        svg_code = vrvToolkit.renderData(mei_string, options=verovio_options)
        logging.info("Verovio rendering successful.")
        display(HTML(svg_code)) # Display in notebook
    except ET.ParseError as e:
        logging.error(f"Generated MEI string is not valid XML: {e}")
        logging.error(f"Problematic MEI string snippet:\n{mei_string[:500]}...")
        print("\n--- ERROR: Could not render invalid MEI ---")
        return None # Return None on error
    except Exception as e:
        logging.error(f"Error during Verovio rendering: {e}", exc_info=True)
        print("\n--- ERROR: Verovio rendering failed ---")
        return None # Return None on error

    # Save the MEI file (if requested)
    if save_mei:
        logging.info(f"Saving MEI string to {mei_save_path}...")
        try:
            with open(mei_save_path, 'w', encoding='utf-8') as f: f.write(mei_string)
            logging.info(f"MEI file saved successfully: {mei_save_path}")
        except IOError as e:
            logging.error(f"Error saving MEI file to {mei_save_path}: {e}")

    # Save the SVG file (if requested and SVG was generated)
    if save_svg and svg_code:
        logging.info(f"Saving rendered SVG to {svg_save_path}...")
        try:
            with open(svg_save_path, 'w', encoding='utf-8') as f: f.write(svg_code)
            logging.info(f"SVG file saved successfully: {svg_save_path}")
        except IOError as e:
            logging.error(f"Error saving SVG file to {svg_save_path}: {e}")
    elif save_svg and not svg_code:
        logging.warning(f"SVG file not saved to {svg_save_path} because SVG code was not generated.")

    return svg_code # Return the SVG code so it can be used by other functions if needed


# --- Main Processing Orchestration Functions ---

def process_and_visualize(
    data: Union[str, pd.DataFrame],
    resolution: str = 'half_tone_deviation',
    clef_config: str = 'GF',
    note_order: str = 'original',
    display_cent_deviation: bool = True,
    include_natural_accidentals: bool = True,
    display_mode: str = 'sequence',
    display_df: bool = True,
    save_mei: bool = False,
    mei_save_path: str = 'output.mei', # Renamed
    save_svg: bool = False,          # New
    svg_save_path: str = 'output.svg'  # New
):
    """
    Processes pitch data, sorts it, and visualizes as musical notation
    either sequentially or as a chord.
    """
    # --- Input Validation ---
    if resolution not in ['half_tone', 'quarter_tone', 'half_tone_deviation', 'eighth_tone']:
         raise ValueError("Invalid resolution. Choose 'half_tone', 'quarter_tone', 'eighth_tone', or 'half_tone_deviation'.")
    if clef_config not in ['G', 'F', 'GF']: raise ValueError("Invalid clef_config.")
    if note_order not in ['original', 'ascending', 'descending']: raise ValueError("Invalid note_order.")
    if display_mode not in ['sequence', 'chord']: raise ValueError("Invalid display_mode. Choose 'sequence' or 'chord'.")

    # --- 1. Load Data ---
    df_orig = _load_dataframe_from_input(data)
    if 'Frequency (Hz)' not in df_orig.columns:
        raise ValueError("Input data must contain a 'Frequency (Hz)' column.")

    # --- 2. Convert Frequencies & Prepare Note Data ---
    logging.info(f"Converting frequencies (resolution: '{resolution}')...")
    p_names_orig, m_vals_orig, c_devs_orig = convert_frequencies(df_orig)
    notes_data_orig = [pitch_name_and_deviation_to_step_octave_alter(pn, cd, resolution) for pn, cd in zip(p_names_orig, c_devs_orig)]

    # --- 3. Combine and Sort Data for MEI ---
    # Filter out None values before zipping for sorting
    combined_data_valid = []
    for i in range(len(m_vals_orig)):
        if m_vals_orig[i] is not None and notes_data_orig[i][0] is not None: # Check MIDI and step
            combined_data_valid.append((m_vals_orig[i], c_devs_orig[i] if c_devs_orig[i] is not None else 0, notes_data_orig[i], i))
    
    logging.info(f"Applying note order: '{note_order}'...")
    sorted_data_for_mei = []
    if not combined_data_valid:
        logging.warning("No valid notes found for sorting/MEI generation.")
    elif note_order == 'ascending':
        sorted_data_for_mei = sorted(combined_data_valid, key=lambda x: (x[0], x[1])) # Sort by MIDI, then deviation
    elif note_order == 'descending':
        sorted_data_for_mei = sorted(combined_data_valid, key=lambda x: (x[0], x[1]), reverse=True)
    else: # 'original'
        sorted_data_for_mei = combined_data_valid

    midi_values_mei, cent_deviations_mei, notes_data_mei = [], [], []
    if sorted_data_for_mei:
        # Unzip, ensuring original indices are not part of the MEI data lists
        temp_m, temp_c, temp_n, _ = zip(*sorted_data_for_mei)
        midi_values_mei = list(temp_m)
        cent_deviations_mei = list(temp_c)
        notes_data_mei = list(temp_n)


    # --- 4. Handle Deviation Display for Chord Mode ---
    effective_display_cent_deviation = display_cent_deviation
    if display_mode == 'chord':
        if display_cent_deviation and resolution == 'half_tone_deviation':
            logging.warning("Cent deviation display with <fing> is typically off for 'chord' mode for clarity. It will be disabled unless explicitly handled by MEI options.")
        effective_display_cent_deviation = False # Usually forced disable for chords to avoid clutter

    # --- 5. Create MEI String ---
    logging.info(f"Creating MEI string (clef: '{clef_config}', mode: '{display_mode}', display cents: {effective_display_cent_deviation})...")
    mei_string = ""
    if notes_data_mei: # Check if there's anything to render
        mei_string = create_mei_string(
            notes_data_mei, midi_values_mei, cent_deviations_mei,
            clef_config, resolution,
            include_natural_accidentals,
            effective_display_cent_deviation, # Pass the potentially modified flag
            display_mode
        )
    else:
        logging.warning("MEI string generation skipped as no valid/sortable notes were found.")

    # --- 6. Display DataFrame (Original Order) ---
    if display_df:
        logging.info("Displaying processed data DataFrame (original order)...")
        df_display_data = {
            'Frequency (Hz)': df_orig['Frequency (Hz)'].tolist(),
            'MIDI Value': ["N/A" if mv is None else mv for mv in m_vals_orig],
            'Cent Deviation': ["N/A" if cd is None else (f"+{cd}" if cd >= 0 else str(cd)) for cd in c_devs_orig],
            'Pitch Name': p_names_orig,
            'MEI Step': ["N/A" if n_info is None or n_info[0] is None else n_info[0] for n_info in notes_data_orig],
            'MEI Octave': ["N/A" if n_info is None or n_info[1] is None else n_info[1] for n_info in notes_data_orig],
            'MEI Alter': ["N/A" if n_info is None or n_info[2] is None else n_info[2] for n_info in notes_data_orig],
        }
        # Ensure all lists have same length for DataFrame creation
        max_len = len(df_orig)
        for k in df_display_data:
            current_len = len(df_display_data[k])
            if current_len < max_len:
                df_display_data[k].extend(["N/A"] * (max_len - current_len))
            elif current_len > max_len: # Should not happen if processing is row-wise
                 df_display_data[k] = df_display_data[k][:max_len]
        
        display(pd.DataFrame(df_display_data))

    # --- 7. Render MEI using Verovio & Save ---
    verovio_options = {
        "inputFormat": "mei", "scale": 50, "adjustPageHeight": False,
        "pageMarginBottom": 200, "noHeader": True, "border": 0,
        "pageHeight": 20000, "pageWidth": 1000, "staffLineWidth": 1.5,
        "systemDividerLineWidth": 1.5, "breaks": "none", "font": "Bravura",
        "fingering": effective_display_cent_deviation,
        "svgHtmlClipPaths": True, "justifyVertically": True,
    }
    if display_mode == 'chord':
         verovio_options['scale'] = 40

    _render_and_save_mei(
        mei_string,
        verovio_options,
        save_mei,
        mei_save_path, # Pass renamed parameter
        save_svg=save_svg, # Pass new parameter
        svg_save_path=svg_save_path # Pass new parameter
    )

def process_temporal_chords(
    data: Union[str, pd.DataFrame],
    freq_col: str = 'freq_start',
    time_start_col: str = 'time_start',
    time_stop_col: str = 'time_stop',
    resolution: str = 'quarter_tone',
    clef_config: str = 'GF',
    note_order_within_chord: str = 'ascending',
    include_natural_accidentals: bool = True,
    measure_represents_sec: float = 0.5,
    save_mei: bool = False,
    mei_save_path: str = 'output_temporal.mei', # Renamed
    save_svg: bool = False,                   # New
    svg_save_path: str = 'output_temporal.svg', # New
    display_df: bool = True
):
    """
    Processes time-structured data, visualizing simultaneous frequencies as chords.
    ...
    Parameters:
        ...
        measure_represents_sec: The visual duration (in seconds) that each MEI measure
                                (containing a chord or padding) should represent.
    """
    # --- Input Validation ---
    if resolution not in ['half_tone', 'quarter_tone', 'eighth_tone', 'half_tone_deviation']:
         if resolution == 'half_tone_deviation':
             logging.warning("Using 'half_tone_deviation' for temporal chords; deviations calculated but not displayed on score.")
    if clef_config not in ['G', 'F', 'GF']: raise ValueError("Invalid clef_config.")
    if note_order_within_chord not in ['original', 'ascending', 'descending']: raise ValueError("Invalid note_order_within_chord.")
    if not isinstance(measure_represents_sec, (int, float)) or measure_represents_sec <= 0:
        raise ValueError("measure_represents_sec must be a positive number.")


    # ... (1. Load Data - no change)
    df_input = _load_dataframe_from_input(data)
    required_cols = [freq_col, time_start_col, time_stop_col]
    if not all(col in df_input.columns for col in required_cols):
        raise ValueError(f"Input data must contain columns: {required_cols}")
    try:
        df_input[freq_col] = pd.to_numeric(df_input[freq_col], errors='coerce')
        df_input[time_start_col] = pd.to_numeric(df_input[time_start_col], errors='coerce')
        df_input[time_stop_col] = pd.to_numeric(df_input[time_stop_col], errors='coerce')
    except Exception as e:
         raise ValueError(f"Could not convert frequency or time columns to numeric: {e}")

    # ... (2. Calculate Pitch Info - no change)
    df_processed = df_input.rename(columns={freq_col: 'Frequency (Hz)'})
    p_names, m_vals, c_devs = convert_frequencies(df_processed)
    notes_data = [pitch_name_and_deviation_to_step_octave_alter(pn, cd, resolution) for pn, cd in zip(p_names, c_devs)]
    df_processed['pitch_name'] = p_names
    df_processed['midi_value'] = m_vals
    df_processed['cent_deviation'] = c_devs
    df_processed['note_info'] = notes_data
    
    # ... (3. Filter invalid pitch data and Group by Time Window - no change in this part's logic)
    original_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=['midi_value', 'note_info', time_start_col, time_stop_col])
    df_processed = df_processed[df_processed['note_info'].apply(lambda x: x is not None and all(val is not None for val in x))]
    
    filtered_rows = len(df_processed)
    if original_rows > filtered_rows:
        logging.warning(f"Removed {original_rows - filtered_rows} rows with invalid pitch/time data before grouping.")
    if df_processed.empty:
        logging.error("No valid pitch data remaining. Cannot generate MEI.")
        return

    grouped = df_processed.groupby([time_start_col, time_stop_col])
    time_groups_for_mei = [] 

    for time_key, group_df in grouped:
        valid_group_df = group_df.dropna(subset=['midi_value', 'note_info'])
        valid_group_df = valid_group_df[valid_group_df['note_info'].apply(lambda x: x is not None and all(v is not None for v in x))]

        if valid_group_df.empty: continue

        notes_list = valid_group_df['note_info'].tolist()
        midis_list = valid_group_df['midi_value'].astype(int).tolist()
        cents_list = valid_group_df['cent_deviation'].astype(int).tolist()
        
        combined_notes_for_chord = list(zip(notes_list, midis_list, cents_list))
        if note_order_within_chord == 'ascending':
            combined_notes_for_chord.sort(key=lambda x: (x[1], x[2])) 
        elif note_order_within_chord == 'descending':
            combined_notes_for_chord.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        if combined_notes_for_chord:
            s_notes, s_midis, s_cents = zip(*combined_notes_for_chord)
            time_groups_for_mei.append((time_key, list(s_notes), list(s_midis), list(s_cents)))
        
    time_groups_for_mei.sort(key=lambda x: x[0][0])
    logging.info(f"Found {len(time_groups_for_mei)} distinct time groups for MEI.")

    # ... (4. Display Grouped Data Summary - no change)
    if display_df and time_groups_for_mei:
        summary_data = []
        for i, (tk, notes, midis, _) in enumerate(time_groups_for_mei):
             summary_data.append({
                 'Group (Measure)': i+1, 'Time Start': tk[0], 'Time Stop': tk[1],
                 'Num Notes': len(notes), 'MIDI Values': sorted(midis),
             })
        display(pd.DataFrame(summary_data))
    elif display_df:
        logging.info("No time groups to display in summary DataFrame.")


    # --- 5. Create Temporal MEI String ---
    mei_string = ""
    if time_groups_for_mei:
        mei_string = create_temporal_mei_string(
            time_groups_for_mei,
            clef_config,
            resolution, # resolution_type was renamed to resolution
            include_natural_accidentals,
            measure_represents_sec=measure_represents_sec # <-- PASS THE PARAMETER
        )
    else:
        logging.warning("MEI string generation skipped as no valid time groups were found.")

    # --- 6. Render MEI using Verovio & Save ---
    verovio_options_temporal = {
        "inputFormat": "mei", "scale": 45, "adjustPageHeight": True,
        "noHeader": True, "border": 0,
        # "pageHeight": 15000, # Optional if adjustPageHeight is True and breaks: none
        "pageWidth": 1800, "staffLineWidth": 1.5, "systemDividerLineWidth": 1.5,
        "breaks": "none",  # Keep as "none" for single long system
        "font": "Bravura", "fingering": False,
        "svgHtmlClipPaths": True, "justifyVertically": True, "pageMarginBottom": 200,
    }
    _render_and_save_mei(
        mei_string,
        verovio_options_temporal,
        save_mei,
        mei_save_path, # Pass renamed parameter
        save_svg=save_svg, # Pass new parameter
        svg_save_path=svg_save_path # Pass new parameter
    )