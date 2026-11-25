"""Install only new camera config files; never edit URDF, teachers or presets."""
import argparse,hashlib,json,pathlib,shutil,xml.etree.ElementTree as ET

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=pathlib.Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();here=pathlib.Path(__file__).resolve().parent;c=json.loads((here/'hu_d04_torso_camera_contract.json').read_text());root=a.bundle.resolve();urdf=root/'asset/urdf/HU_D04_01.urdf';mesh=root/'asset/meshes/HU_D04_01'
 assert hashlib.sha256(urdf.read_bytes()).hexdigest()==c['training_urdf_sha256'],'Training URDF changed; re-audit before installation.'
 links={x.attrib['name'] for x in ET.fromstring(urdf.read_bytes()).findall('link')};assert c['parent_link'] in links and set(c['ray_cast_bodies'])<=links
 for name,digest in c['mesh_sha256'].items():assert hashlib.sha256((mesh/name).read_bytes()).hexdigest()==digest,name
 dest=root/'runtime/holosoma/src/holosoma/holosoma/sensors/warp/camera_config';assert (dest/'d435i_depth_config.py').is_file()
 names=['hu_d04_torso_depth_config.py','hu_d04_torso_camera_contract.json']
 if not a.check_only:
  for n in names:
   target=dest/n
   if target.exists():assert target.read_bytes()==(here/n).read_bytes(),'Existing overlay differs; refusing to overwrite.'
  for n in names:
   if not (dest/n).exists():shutil.copy2(here/n,dest/n)
   assert (dest/n).read_bytes()==(here/n).read_bytes()
 print(json.dumps(dict(status='CHECK_ONLY_PASSED' if a.check_only else 'CAMERA_MODULE_INSTALLED_NOT_TRAINING_INTEGRATED',destination=str(dest),checked_meshes=len(c['mesh_sha256']),preset_modified=False,urdf_modified=False,new_gpu_tasks=0,native_render_verified=False)))

if __name__=='__main__':main()