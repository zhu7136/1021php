"""CPU-only URDF forward kinematics and geometric Jacobians; no simulator imports."""
from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT/'flux/holosoma_overlay/data/robots/hu_d04/HU_D04_01_train.urdf'


def robot_config():
    values = {}
    tree = ast.parse((ROOT/'flux/holosoma_overlay/config_values/robot.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith('_HU_D04_'):
                    values[target.id] = ast.literal_eval(node.value)
    return values


class Model:
    def __init__(self, path=URDF, names=None):
        self.names = names or robot_config()['_HU_D04_DOF_NAMES']
        self.indices = {n:i for i,n in enumerate(self.names)}
        xml = ET.parse(path).getroot()
        pending = []
        for node in xml.findall('joint'):
            origin = node.find('origin'); axis = node.find('axis'); limit = node.find('limit')
            xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            a = np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0,0])
            a = a / np.linalg.norm(a)
            pending.append(dict(name=node.get('name'), kind=node.get('type'), parent=node.find('parent').get('link'), child=node.find('child').get('link'), xyz=xyz, R=Rotation.from_euler('xyz',rpy).as_matrix(), axis=a, limit={} if limit is None else {k:float(v) for k,v in limit.attrib.items()}))
        roots = {l.get('name') for l in xml.findall('link')} - {j['child'] for j in pending}
        if len(roots)!=1:
            raise ValueError(f'Expected one URDF root: {roots}')
        self.root = roots.pop(); self.joints=[]; self.ancestors={self.root:[]}
        while pending:
            layer=[j for j in pending if j['parent'] in self.ancestors]
            if not layer:
                raise ValueError('Disconnected/cyclic URDF')
            for j in layer:
                self.ancestors[j['child']] = self.ancestors[j['parent']] + ([self.indices[j['name']]] if j['name'] in self.indices else [])
                self.joints.append(j);pending.remove(j)
        byname = {j['name']:j for j in self.joints}
        self.lower = np.array([byname[n]['limit']['lower'] for n in self.names])
        self.upper = np.array([byname[n]['limit']['upper'] for n in self.names])
        self.velocity = np.array([byname[n]['limit']['velocity'] for n in self.names])

    def forward(self, q, root_pos, root_quat):
        """root_quat is wxyz. Returned angular Jacobians use world coordinates."""
        q=np.asarray(q);n=len(q)
        positions={self.root:root_pos};rotations={self.root:Rotation.from_quat(root_quat[:,[1,2,3,0]]).as_matrix()}
        joint_positions=np.zeros((n,len(self.names),3));joint_axes=np.zeros_like(joint_positions)
        for j in self.joints:
            p=positions[j['parent']]+np.einsum('nij,j->ni',rotations[j['parent']],j['xyz'])
            r=rotations[j['parent']]@j['R']
            if j['kind'] in ('revolute','continuous'):
                if j['name'] in self.indices:
                    i=self.indices[j['name']];angle=q[:,i]
                    joint_positions[:,i]=p;joint_axes[:,i]=np.einsum('nij,j->ni',r,j['axis'])
                else:
                    angle=np.zeros(n)
                r=r@Rotation.from_rotvec(angle[:,None]*j['axis']).as_matrix()
            elif j['kind']!='fixed':
                raise ValueError(f'Unsupported joint type: {j["kind"]}')
            positions[j['child']]=p;rotations[j['child']]=r
        return positions,rotations,joint_positions,joint_axes

    def jacobian(self, body, columns, positions, joint_positions, joint_axes):
        linear=np.cross(joint_axes[:,columns],positions[body][:,None,:]-joint_positions[:,columns])
        angular=joint_axes[:,columns].copy()
        active=np.isin(columns,self.ancestors[body])
        linear[:,~active]=0;angular[:,~active]=0
        return linear.transpose(0,2,1),angular.transpose(0,2,1)

    def chain_length(self, start, end):
        parents={j['child']:j for j in self.joints}
        length=0.
        while end!=start:
            joint=parents[end]
            length+=float(np.linalg.norm(joint['xyz']))
            end=joint['parent']
        return length