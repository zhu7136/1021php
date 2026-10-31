"""Hu-D04 chest camera: explicit URDF-parent geometry, not a hardware calibration.

The official camera joints are XML comments. They are reference mounting
information, not active simulator sensors. Real intrinsics still need calibration.
"""
from pathlib import Path
import math,os
import xml.etree.ElementTree as ET
import numpy as np

def rotation(rpy):
 r,p,y=rpy;cr,sr=math.cos(r),math.sin(r);cp,sp=math.cos(p),math.sin(p);cy,sy=math.cos(y),math.sin(y)
 return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])@np.array([[cp,0,sp],[0,1.,0],[-sp,0,cp]])@np.array([[1.,0,0],[0,cr,-sr],[0,sr,cr]])

def transform(origin):
 result=np.eye(4)
 if origin is not None:
  result[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
  result[:3,:3]=rotation(np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
 return result

def parse_reference(urdf):
 parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
 model=ET.parse(urdf,parser=parser).getroot()
 active_joints=model.findall('joint')
 mount=None
 for node in model.iter():
  if node.tag is ET.Comment and '<joint name="waist_camera_joint"' in (node.text or ''):
   fragment=ET.fromstring('<reference>'+node.text+'</reference>')
   mount=fragment.find("joint[@name='waist_camera_joint']")
 if mount is None:raise ValueError('Official commented chest mount reference missing')
 parent=mount.find('parent').get('link')
 if parent!='waist_pitch_link':raise ValueError('Chest parent differs from chosen contract')
 origin=mount.find('origin');xyz=np.fromstring(origin.get('xyz'),sep=' ')
 urdf_rpy=np.fromstring(origin.get('rpy'),sep=' ')
 # URDF mount's positive Z boresight is 90deg + downward pitch about Y.
 # Warp optical rays use +Z; its data-frame correction maps +Z to body +X.
 pitch_degrees=math.degrees(urdf_rpy[1])-90.
 if np.linalg.norm(urdf_rpy[[0,2]])>1e-8 or not 34.9<pitch_degrees<35.1:
  raise ValueError('Unexpected chest rotation convention; explicit re-audit required')
 child_to_joint={j.find('child').get('link'):j for j in active_joints}
 chain=[];link=parent
 while link!='base_link':
  if link not in child_to_joint or link in [x[0] for x in chain]:raise ValueError('Missing/cyclic parent chain')
  joint=child_to_joint[link];chain.append((link,transform(joint.find('origin'))));link=joint.find('parent').get('link')
 base_to_parent=np.eye(4)
 for _,matrix in reversed(chain):base_to_parent=base_to_parent@matrix
 point=(base_to_parent@np.r_[xyz,1])[:3]
 render_rotation=rotation(np.radians([0,pitch_degrees,0]))@rotation(np.radians([-90,0,-90]))
 return {'parent':parent,'offset_xyz_m':xyz.tolist(),'urdf_mount_rpy_rad':urdf_rpy.tolist(),
  'renderer_mount_rpy_deg':[0.,pitch_degrees,0.],'renderer_optical_correction_deg':[-90.,0.,-90.],
  'base_to_parent_zero_pose_xyz_m':base_to_parent[:3,3].tolist(),
  'camera_base_link_zero_pose_xyz_m':point.tolist(),'boresight_parent_xyz':render_rotation[:,2].tolist(),
  'urdf_positive_z_boresight':rotation(urdf_rpy)[:,2].tolist(),
  'active_camera_joint':any(j.get('name')=='waist_camera_joint' for j in active_joints),
  'source_kind':'commented official URDF mount; zero-joint-angle kinematics; not hardware calibration'}

def visual_meshes(asset_root):
 root=Path(asset_root).resolve(strict=True);model=ET.parse(root/'urdf/HU_D04_01.urdf').getroot();meshes={}
 for link in model.findall('link'):
  visuals=link.findall('visual');mesh_visuals=[v for v in visuals if v.find('geometry/mesh') is not None]
  if not mesh_visuals:continue
  if len(mesh_visuals)!=1:raise ValueError('Multiple visual meshes need explicit transform baking')
  visual=mesh_visuals[0];mesh=visual.find('geometry/mesh')
  if not np.allclose(transform(visual.find('origin')),np.eye(4),atol=1e-12):
   raise ValueError('Visual origin requires mesh transform baking: '+link.get('name'))
  scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
  if not np.allclose(scale,[1,1,1]):raise ValueError('Non-unit visual mesh scale')
  name=mesh.get('filename')
  if '/meshes/' not in name:raise ValueError('Unrecognized mesh URI')
  relative=name.split('/meshes/',1)[1];path=root/'meshes'/relative
  if not path.resolve(strict=True).is_relative_to(root/'meshes'):raise ValueError('Escaping mesh path')
  meshes[link.get('name')]=relative
 if not {'base_link','waist_pitch_link','head_yaw_link','head_pitch_link'}<=set(meshes):
  raise ValueError('Missing physical root/torso/head meshes')
 return meshes

class HuD04ChestDepthConfig:
 # Nominal D435-like sampling retained from the existing distillation
 # pipeline; this is NOT a claim of measured Hu-D04 camera intrinsics.
 num_sensors=1; sensor_type='camera';width=106;height=60;horizontal_fov_deg=89.5
 min_range=0.3;max_range=3.;crop_top=2;crop_bottom=0;crop_left=4;crop_right=4
 dynamic_meshes=True;randomize_placement=False
 calculate_depth=True;return_pointcloud=False;pointcloud_in_world_frame=False;segmentation_camera=False
 add_offpath_obstacle=False;offpath_obstacle_meshes_root=None;offpath_obstacle_bodies={}
 def __init__(self,asset_root=None):
  if asset_root is None:asset_root=os.environ.get('HUD04_ASSET_ROOT')
  if not asset_root:raise ValueError('Explicit HUD04_ASSET_ROOT required; no G1 fallback')
  root=Path(asset_root).resolve(strict=True)
  reference=parse_reference(root/'urdf/HU_D04_01.urdf')
  self.base_link_frame={'cam_front_depth':reference['parent']}
  self.offset={'cam_front_depth':{'offset_pos':tuple(reference['offset_xyz_m']),'offset_rot':tuple(reference['renderer_mount_rpy_deg'])}}
  self.offset_rot_base=reference['renderer_optical_correction_deg']
  self.asset_meshes_root=str(root/'meshes');self.ray_cast_bodies=visual_meshes(root)
  self.mount_reference=reference