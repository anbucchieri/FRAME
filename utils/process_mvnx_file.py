import os 
import csv
import xml.etree.ElementTree as ET
import xacro

XACRO_NS = 'http://www.ros.org/wiki/xacro'

def prefix_xacro_macros(xacro_root):
    '''
    The base xacro files invoke their macros WITHOUT the 'xacro:' namespace
    prefix, e.g. <box_inertia .../> instead of <xacro:box_inertia .../>.
    Modern xacro only expands elements that live in the xacro namespace, so those
    un-prefixed calls are passed through verbatim and end up as invalid URDF:
        - <inertial> ends up containing <box_inertia>/<cylinder_inertia> instead of
          the <inertia> tag the macro should have produced
          -> "Inertial element must have inertia element"
        - <virtual_body name="body_xxx"/> never becomes a real <link>, so any joint
          that refers to it fails
          -> "parent link [body_xxx] of joint [...] not found"

    This restores the prefix on every un-prefixed call of a macro that is actually
    defined in the document (auto-detected, so new macros are handled too), and
    removes stray text nodes that are just a leftover '>' character.

    Must be called on the ElementTree root AFTER ET.register_namespace('xacro', ...)
    and BEFORE the tree is written back to disk.
    '''
    ns = '{%s}' % XACRO_NS
    # Discover every macro name defined via <xacro:macro name="...">
    macro_names = {el.get('name') for el in xacro_root.iter(ns + 'macro') if el.get('name')}
    for elem in xacro_root.iter():
        if not isinstance(elem.tag, str):
            continue                       # skip comments / processing instructions
        if '}' not in elem.tag and elem.tag in macro_names:
            elem.tag = ns + elem.tag       # un-prefixed macro call -> add xacro: prefix
        # Scrub stray '>' that leaked in as text from the source file
        if elem.text and elem.text.strip() == '>':
            elem.text = None
        if elem.tail and elem.tail.strip() == '>':
            elem.tail = None
    return macro_names

def write_Xsens_movement_data_to_csv(mvnx_file):
    '''
    Read an Xsens MVNX motion file and create a CSV file containing the position, orientation (quaternion) 
    and center-of-mass data for each Xsens segment, for all time frames (tpose and identity frames excluded). 
    The COM data includes position (p_COM: first three values), velocity (v_COM: next three values) 
    and acceleration (a_COM: last three values).
    
    CSV file is created in the same folder as the MVNX file.
    
    Input: path to Xsens MVNX file
    Output: 
        path to created CSV file
        Xsens capture rate
    '''
    import os
    import csv
    import xml.etree.ElementTree as ET

    if not os.path.isfile(mvnx_file):
        print(f"MVNX File not found: {mvnx_file}")
        return

    ns = "{http://www.xsens.com/mvn/mvnx}"
    try:
        tree = ET.parse(mvnx_file)
        root = tree.getroot()
    except OSError as e:
        raise e

    # Extract frame rate from Xsens file (in Hz)
    for subject in root.iter(ns + 'subject'):
        frame_rate = int(subject.get('frameRate'))
    
    # Extract label names for segments (quaternions and positions)
    label_names = ["timestamp"]
    label_orientation = []
    label_position = []
    label_omega= []
    label_omega_dot = []
    for label in root.iter(ns + 'segment'):
        label_orientation.extend([
            label.get("label") + "_w_q", 
            label.get("label") + "_x_q", 
            label.get("label") + "_y_q", 
            label.get("label") + "_z_q"
        ])
        label_position.extend([
            label.get("label") + "_X", 
            label.get("label") + "_Y", 
            label.get("label") + "_Z"
        ])
    label_names.extend(label_orientation)
    label_names.extend(label_position)
    
    # Add labels for center-of-mass data:
    # COM position: p_COM, COM velocity: v_COM, COM acceleration: a_COM (each with three components)
    label_names.extend([
        "COM_pX", "COM_pY", "COM_pZ",
        "COM_vX", "COM_vY", "COM_vZ",
        "COM_aX", "COM_aY", "COM_aZ"
    ])
    
    # Add labels for angular velocity and acceleration:
    for label in root.iter(ns + 'segment'):
        label_omega.extend([
            label.get("label") + "_x_omega", 
            label.get("label") + "_y_omega", 
            label.get("label") + "_z_omega"
        ])
        label_omega_dot.extend([
            label.get("label") + "_x_omega_dot", 
            label.get("label") + "_y_omega_dot", 
            label.get("label") + "_z_omega_dot"
        ])
    label_names.extend(label_omega)
    label_names.extend(label_omega_dot)

    def extraction(data_type):
        extracted_data = []
        for data in root.iter(ns + data_type):
            extracted_data.append([float(i) for i in data.text.split()])
        return extracted_data

    # Extract all position, orientation, and centerOfMass data 
    segment_orientation = extraction('orientation')
    segment_position = extraction('position')
    com_data = extraction('centerOfMass')
    segment_angular_velocity = extraction('angularVelocity')
    segment_angular_acceleration = extraction('angularAcceleration')
    
    # Remove the first 3 elements corresponding to 'identity', 'tpose', and 'tpose-isb' frames
    segment_orientation = segment_orientation[3:]
    segment_position = segment_position[3:]

    # Some exports omit angularVelocity / angularAcceleration entirely (several RooiBoS subjects
    # do). The loop below indexes them unconditionally, so a missing field used to abort the whole
    # recording with "IndexError: list index out of range" -- losing the capture over two columns
    # that nothing reads: joint velocity is differentiated from the ANGLES downstream
    # (frame_core.signal.derivative) and ID_Xsens takes its angular velocity from the Pinocchio
    # frame velocities, not from this CSV. So pad them to keep the row width matching the header.
    n_frames_expected = min(len(segment_orientation), len(segment_position), len(com_data))
    n_seg = len(label_omega) // 3
    if len(segment_angular_velocity) < n_frames_expected:
        segment_angular_velocity = (list(segment_angular_velocity)
                                    + [[0.0] * (3 * n_seg)] * n_frames_expected)
    if len(segment_angular_acceleration) < n_frames_expected:
        segment_angular_acceleration = (list(segment_angular_acceleration)
                                        + [[0.0] * (3 * n_seg)] * n_frames_expected)

    # Write to CSV
    output_csv_filename = mvnx_file.split(".")[0] + "_com_mov.csv"
    with open(output_csv_filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(label_names)

        for frame in root.iter(ns + 'frame'):
            if frame.get('type') == "normal":
                frame_index = int(frame.get('index')) 
                frame_timestamp = int(frame.get('time')) / 1000.  # Convert ms to sec

                # Ensure that the frame index is within the range of our extracted data arrays
                if (frame_index >= len(segment_orientation) or 
                    frame_index >= len(segment_position) or 
                    frame_index >= len(com_data)):
                    raise ValueError("Wrong frame index")
                
                # Slice COM data into position, velocity, and acceleration components
                com_values = com_data[frame_index]
                p_COM = com_values[0:3]    # First three values
                v_COM = com_values[3:6]    # Next three values
                a_COM = com_values[6:9]    # Last three values

                # Compose the CSV row: timestamp, orientations, positions, and COM data
                line = (
                    [frame_timestamp] + 
                    segment_orientation[frame_index] + 
                    segment_position[frame_index] + 
                    p_COM + v_COM + a_COM +
                    segment_angular_velocity[frame_index] +
                    segment_angular_acceleration[frame_index]
                )
                wr.writerow(line)

    return output_csv_filename, frame_rate

def get_body_dimensions_from_Xsens(mvnx_file):
    '''
    Read an Xsens MVNX file and return the 3D coordinates of all anatomical points (defined in Xsens)
    for all Xsens segments.
    Input: path to Xsens MVNX file
    Output: list of anatomical points, per segment, with their coordinates
    '''
    if not os.path.isfile(mvnx_file):
        print(f"MVNX file not found: {mvnx_file}")
        return

    ns = "{http://www.xsens.com/mvn/mvnx}"
    try:
        tree = ET.parse(mvnx_file)
        root = tree.getroot()
    except OSError as e:
        raise e

    # Extract anatomical points of segments
    anatomical_points = dict()
    for segment in root.iter(ns + 'segment'):
        segment_name = segment.get("label")
        anatomical_points[segment_name] = dict()
        for point in segment[0]:
            anatomical_points[segment_name][point.get("label")] = [float(point[0].text.split(" ")[0]), float(point[0].text.split(" ")[1]), float(point[0].text.split(" ")[2])]

    return anatomical_points

def create_urdf_from_Xsens_new(mvnx_file, human_height, human_mass, urdf_folder, xacro_base_file):
    '''
    From a human description in a xacro file and from an Xsens MVNX motion file, 
    create a URDF model corresponding to the xacro description but with body dimensions scaled from 
    the Xsens data.
    Input:
        mvnx_file: path to the Xsens MVNX file (its segment dimensions scale the model)
        human_height: body height of the subject [m]
        human_mass: body mass of the subject [kg]
        urdf_folder: folder the scaled .xacro and .urdf are written to (with a trailing '/')
        xacro_base_file: the DHM template (models/dhm_new.xacro)
    Output: path to the created URDF file, '<urdf_folder><mvnx stem>.urdf'
    '''
    # Read segment dimensions from Xsens file
    anatomical_points = get_body_dimensions_from_Xsens(mvnx_file)
    offset_force_plate = 0 # [m]
    
    # Calculate body segments dimensions
    # ---- Trunk ---- #
    S5_L5 = abs(anatomical_points['Pelvis']['jL5S1'][2] - anatomical_points['Pelvis']['pHipOrigin'][2]) 
    L5_T8 = abs(anatomical_points['L5']['jL5S1'][2] - anatomical_points['L5']['jL4L3'][2]) + abs(anatomical_points['L3']['jL4L3'][2] - anatomical_points['L3']['jL1T12'][2]) + abs(anatomical_points['T12']['jL1T12'][2] - anatomical_points['T12']['jT9T8'][2])
    T8_C7 = abs(anatomical_points['T8']['jT9T8'][2] - anatomical_points['T8']['jT1C7'][2])
    C7_C1 = abs(anatomical_points['Neck']['jT1C7'][2] - anatomical_points['Neck']['jC1Head'][2])
    C1_headtop = abs(anatomical_points['Head']['jC1Head'][2] - anatomical_points['Head']['pTopOfHead'][2])
    head_radius = abs(anatomical_points['Head']['pTopOfHead'][2] - anatomical_points['Head']['pCenterOfHead'][2])

    # ---- Right arm ----#
    r_T8_CSJheight = abs(anatomical_points['T8']['jT9T8'][2] - anatomical_points['T8']['jRightT4Shoulder'][2]) 
    r_spine_CSJ = abs(anatomical_points['T8']['jT9T8'][1] - anatomical_points['T8']['jRightT4Shoulder'][1]) 
    r_shoulderWidth = abs(anatomical_points['RightShoulder']['jRightT4Shoulder'][1] - anatomical_points['RightShoulder']['jRightShoulder'][1])
    r_armLength = abs(anatomical_points['RightUpperArm']['jRightShoulder'][1] - anatomical_points['RightUpperArm']['jRightElbow'][1])
    r_forearmLength = abs(anatomical_points['RightForeArm']['jRightElbow'][1] - anatomical_points['RightForeArm']['jRightWrist'][1])
    r_handLength = abs(anatomical_points['RightHand']['jRightWrist'][1] - anatomical_points['RightHand']['pRightTopOfHand'][1])

    # ---- Left arm ----#
    l_T8_CSJheight = abs(anatomical_points['T8']['jT9T8'][2] - anatomical_points['T8']['jLeftT4Shoulder'][2]) 
    l_spine_CSJ = abs(anatomical_points['T8']['jT9T8'][1] - anatomical_points['T8']['jLeftT4Shoulder'][1]) 
    l_shoulderWidth = abs(anatomical_points['LeftShoulder']['jLeftT4Shoulder'][1] - anatomical_points['LeftShoulder']['jLeftShoulder'][1])
    l_armLength = abs(anatomical_points['LeftUpperArm']['jLeftShoulder'][1] - anatomical_points['LeftUpperArm']['jLeftElbow'][1])
    l_forearmLength = abs(anatomical_points['LeftForeArm']['jLeftElbow'][1] - anatomical_points['LeftForeArm']['jLeftWrist'][1])
    l_handLength = abs(anatomical_points['LeftHand']['jLeftWrist'][1] - anatomical_points['LeftHand']['pLeftTopOfHand'][1])

    # ---- Right leg ----#
    r_hipWidth = abs(anatomical_points['Pelvis']['jL5S1'][1] - anatomical_points['Pelvis']['jRightHip'][1])
    r_legLength = abs(anatomical_points['RightUpperLeg']['jRightHip'][2] - anatomical_points['RightUpperLeg']['jRightKnee'][2])
    r_shankLength = abs(anatomical_points['RightLowerLeg']['jRightKnee'][2] - anatomical_points['RightLowerLeg']['jRightAnkle'][2])
    r_footThickness = abs(anatomical_points['RightFoot']['jRightAnkle'][2] - anatomical_points['RightFoot']['pRightHeelFoot'][2]) + offset_force_plate
    r_footFrontLength = abs(anatomical_points['RightFoot']['jRightAnkle'][0] - anatomical_points['RightFoot']['jRightBallFoot'][0]) + abs(anatomical_points['RightToe']['jRightBallFoot'][0] - anatomical_points['RightToe']['pRightToe'][0])
    r_footRearLength = abs(anatomical_points['RightFoot']['pRightHeelFoot'][0] - anatomical_points['RightFoot']['jRightAnkle'][0]) 

    # ---- Left leg ----#
    l_hipWidth = abs(anatomical_points['Pelvis']['jL5S1'][1] - anatomical_points['Pelvis']['jLeftHip'][1])
    l_legLength = abs(anatomical_points['LeftUpperLeg']['jLeftHip'][2] - anatomical_points['LeftUpperLeg']['jLeftKnee'][2])
    l_shankLength = abs(anatomical_points['LeftLowerLeg']['jLeftKnee'][2] - anatomical_points['LeftLowerLeg']['jLeftAnkle'][2])
    l_footThickness = abs(anatomical_points['LeftFoot']['jLeftAnkle'][2] - anatomical_points['LeftFoot']['pLeftHeelFoot'][2])  + offset_force_plate
    l_footFrontLength = abs(anatomical_points['LeftFoot']['jLeftAnkle'][0] - anatomical_points['LeftFoot']['jLeftBallFoot'][0]) + abs(anatomical_points['LeftToe']['jLeftBallFoot'][0] - anatomical_points['LeftToe']['pLeftToe'][0])
    l_footRearLength = abs(anatomical_points['LeftFoot']['pLeftHeelFoot'][0] - anatomical_points['LeftFoot']['jLeftAnkle'][0]) 

    # Replace body dimensions of xacro base by xsens ones
    if not os.path.isfile(xacro_base_file):
        print(f"Xacro file not found: {xacro_base_file}")
        return

    xacro_file = xacro.parse(open(xacro_base_file))
    xacro_xml_string = xacro_file.toxml()
    xacro_root = ET.fromstring(xacro_xml_string)

    ET.register_namespace('xacro', 'http://www.ros.org/wiki/xacro')

    human_bmi = float(human_mass)/(float(human_height)*float(human_height))
    for prop in xacro_root.findall(".//xacro:property", namespaces={'xacro': 'http://www.ros.org/wiki/xacro'}):
        if prop.attrib.get('name') == 'bodyHeight':
            prop.set('value', str(human_height))
        elif prop.attrib.get('name') == 'BMI':
            prop.set('value', str(human_bmi))
        elif prop.attrib.get('name') == 'bodyMass':
            prop.set('value', str(human_mass))
        elif prop.attrib.get('name') == 'S5_L5':
            prop.set('value', str(S5_L5))
        elif prop.attrib.get('name') == 'L5_T8':
            prop.set('value', str(L5_T8))    
        elif prop.attrib.get('name') == 'T8_C7':
            prop.set('value', str(T8_C7))    
        elif prop.attrib.get('name') == 'C7_C1':
            prop.set('value', str(C7_C1))
        elif prop.attrib.get('name') == 'C1_headtop':
            prop.set('value', str(C1_headtop))
        elif prop.attrib.get('name') == 'head_radius':
            prop.set('value', str(head_radius))
        elif prop.attrib.get('name') == 'r_T8_CSJheight':
            prop.set('value', str(r_T8_CSJheight))
        elif prop.attrib.get('name') == 'r_spine_CSJ':
            prop.set('value', str(r_spine_CSJ))
        elif prop.attrib.get('name') == 'r_shoulderWidth':
            prop.set('value', str(r_shoulderWidth))
        elif prop.attrib.get('name') == 'r_spine_shoulder':
            prop.set('value', str(r_shoulderWidth+r_spine_CSJ))
        elif prop.attrib.get('name') == 'r_armLength':
            prop.set('value', str(r_armLength))
        elif prop.attrib.get('name') == 'r_forearmLength':
            prop.set('value', str(r_forearmLength))
        elif prop.attrib.get('name') == 'r_handLength':
            prop.set('value', str(r_handLength))
        elif prop.attrib.get('name') == 'l_T8_CSJheight':
            prop.set('value', str(l_T8_CSJheight))
        elif prop.attrib.get('name') == 'l_spine_CSJ':
            prop.set('value', str(l_spine_CSJ))
        elif prop.attrib.get('name') == 'l_shoulderWidth':
            prop.set('value', str(l_shoulderWidth))
        elif prop.attrib.get('name') == 'l_spine_shoulder':
            prop.set('value', str(l_shoulderWidth+l_spine_CSJ))
        elif prop.attrib.get('name') == 'l_armLength':
            prop.set('value', str(l_armLength))
        elif prop.attrib.get('name') == 'l_forearmLength':
            prop.set('value', str(l_forearmLength))
        elif prop.attrib.get('name') == 'l_handLength':
            prop.set('value', str(l_handLength))
        elif prop.attrib.get('name') == 'r_hipWidth':
            prop.set('value', str(r_hipWidth))
        elif prop.attrib.get('name') == 'r_legLength':
            prop.set('value', str(r_legLength))
        elif prop.attrib.get('name') == 'r_shankLength':
            prop.set('value', str(r_shankLength))
        elif prop.attrib.get('name') == 'r_legLength':
            prop.set('value', str(r_legLength))
        elif prop.attrib.get('name') == 'r_footThickness':
            prop.set('value', str(r_footThickness))
        elif prop.attrib.get('name') == 'r_footFrontLength':
            prop.set('value', str(r_footFrontLength))
        elif prop.attrib.get('name') == 'r_footRearLength':
            prop.set('value', str(r_footRearLength))  
        elif prop.attrib.get('name') == 'l_hipWidth':
            prop.set('value', str(l_hipWidth))
        elif prop.attrib.get('name') == 'l_legLength':
            prop.set('value', str(l_legLength))
        elif prop.attrib.get('name') == 'l_shankLength':
            prop.set('value', str(l_shankLength))
        elif prop.attrib.get('name') == 'l_legLength':
            prop.set('value', str(l_legLength))
        elif prop.attrib.get('name') == 'l_footThickness':
            prop.set('value', str(l_footThickness))
        elif prop.attrib.get('name') == 'l_footFrontLength':
            prop.set('value', str(l_footFrontLength))
        elif prop.attrib.get('name') == 'l_footRearLength':
            prop.set('value', str(l_footRearLength))   

    # ==========================================
    # FIX: Add xacro: prefix to un-prefixed macros
    # (auto-detects macro names and scrubs stray '>'; see prefix_xacro_macros)
    # ==========================================
    prefix_xacro_macros(xacro_root)
    # ==========================================

    
    # Write updated xacro file 
    trial_name = (mvnx_file.split('/')[-1]).split('.')[0]
    scaled_xacro_name = urdf_folder + trial_name + ".xacro"
    tree = ET.ElementTree(xacro_root)
    with open(scaled_xacro_name, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
        
    # Convert xacro to URDF
    scaled_xacro_file = xacro.process_file(scaled_xacro_name)
    urdf_content = scaled_xacro_file.toxml()
    
    urdf_name = urdf_folder  + trial_name + ".urdf"
    with open(urdf_name, "w") as urdf_file:
        urdf_file.write(urdf_content)   

    return urdf_name
