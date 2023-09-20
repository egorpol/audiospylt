import numpy as np
import pandas as pd
from typing import List, Tuple
from IPython.display import display, HTML
import verovio

A4_FREQUENCY = 440

def process_and_visualize(data: [str, pd.DataFrame], save_mei: bool = False, save_path: str = 'output.mei', display_df: bool = True):
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
    notes_and_alters = [pitch_name_and_cent_deviation_to_step_octave_alter(pitch_name, cent_deviation) for pitch_name, cent_deviation in zip(pitch_names, midi_cent_deviations)]
    mei = create_mei_string(notes_and_alters, midi_values.copy(), staff='GF')

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
    htmlCode = vrvToolkit.renderData(mei, options=options)
    display(HTML(htmlCode))

    # 6. Provide an option to save the MEI string to a file
    if save_mei:
        with open(save_path, 'w') as f:
            f.write(mei)
            
# Supporting functions (convert_frequencies, acoustical_pitch_name, midi_value, midi_cent_value, pitch_name_to_step_octave_alter, pitch_name_and_cent_deviation_to_step_octave_alter, create_mei_string) go here

def convert_frequencies(df: pd.DataFrame) -> Tuple[List[str], List[int], List[str]]:
    frequencies = df['Frequency (Hz)'].tolist()
    def acoustical_pitch_name(midi_num: int) -> str:
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        pitch_idx = midi_num % 12
        pitch_name = pitch_names[pitch_idx]
        octave = (midi_num - 12) // 12
        return f"{pitch_name}{octave}"

    def midi_value(log_freq_ratio: float) -> int:
        return int(round(12 * log_freq_ratio + 69))

    def midi_cent_value(log_freq_ratio: float) -> int:
        return int(round(69 * 100 + 1200 * log_freq_ratio))

    log_freq_ratios = [np.log2(freq / A4_FREQUENCY) for freq in frequencies]
    midi_values = [midi_value(log_freq_ratio) for log_freq_ratio in log_freq_ratios]
    midi_cent_values = [midi_cent_value(log_freq_ratio) for log_freq_ratio in log_freq_ratios]
    pitch_names = [acoustical_pitch_name(midi_value) for midi_value in midi_values]
    midi_cent_deviations = [((midi_cent_value - midi_value * 100) + 50) % 100 - 50 for midi_value, midi_cent_value in zip(midi_values, midi_cent_values)]
    midi_cent_deviations = [f'+{x}' if x >= 0 else f'{x}' for x in midi_cent_deviations]

    return pitch_names, midi_values, midi_cent_deviations


def pitch_name_to_step_octave_alter(pitch_name: str) -> Tuple[str, int, int]:
    step = pitch_name[0]
    alter = 1 if '#' in pitch_name else -1 if 'b' in pitch_name else 0
    octave = int(pitch_name[-1])
    return step, octave, alter

def pitch_name_and_cent_deviation_to_step_octave_alter(pitch_name: str, cent_deviation: str) -> Tuple[str, int, str]:
    step = pitch_name[0]
    octave = int(pitch_name[-1])
    deviation = int(cent_deviation)
    alter = 'n'
    if '#' in pitch_name:
        alter = 's'
        if deviation >= 25:
            alter = '3qs'
        elif deviation > 0:
            alter = '1qs'
    elif 'b' in pitch_name:
        alter = 'f'
        if deviation <= -25:
            alter = '3qf'
        elif deviation < 0:
            alter = '1qf'
    else:
        if deviation >= 25:
            alter = '1qs'
        elif deviation > 0:
            alter = '1qs'
        elif deviation <= -25:
            alter = '1qf'
        elif deviation < 0:
            alter = '1qf'
    return step, octave, alter

def create_mei_string(notes_and_alters: List[Tuple[str, int, str]], midi_values: List[int], staff: str) -> str:
    staff_defs = ''
    if 'G' in staff:
        staff_defs += '<staffDef n="1" lines="5" clef.shape="G" clef.line="2"/>\n'
    if 'F' in staff:
        staff_defs += '<staffDef n="2" lines="5" clef.shape="F" clef.line="4"/>\n'
    
    mei = f'''
    <?xml version="1.0" encoding="UTF-8"?>
    <mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.1">
        <music>
            <body>
                <mdiv>
                    <score>
                        <scoreDef keysig="0" mode="major">
                            <staffGrp n="1">
                                {staff_defs}
                            </staffGrp>
                        </scoreDef>
                        <section>
    '''
    for i, ((step, octave, alter), midi_value) in enumerate(zip(notes_and_alters, midi_values)):
        mei += f'<measure n="{i+1}" right="invis">\n'
        if 'G' in staff:
            mei += '<staff n="1">\n<layer n="1">\n'
            if octave >= 4 or (octave == 3 and step == 'B'):
                mei += f'''
                                        <note dur="4" oct="{octave}" pname="{step.lower()}" pnum="{midi_value}" stem.dir="down" accid="{alter}" stem.visible="false"/>
                '''
            mei += '</layer>\n</staff>\n'
        if 'F' in staff:
            mei += '<staff n="2">\n<layer n="1">\n'
            if octave < 4 or (octave == 4 and step == 'C' and alter in ['f', '1qf', '3qf']):
                mei += f'''
                                        <note dur="4" oct="{octave}" pname="{step.lower()}" pnum="{midi_value}" stem.dir="down" accid="{alter}" stem.visible="false"/>
                '''
            mei += '</layer>\n</staff>\n'
        mei += '</measure>\n'
    mei += '''
                        </section>
                    </score>
                </mdiv>
            </body>
        </music>
    </mei>
    '''
    return mei

