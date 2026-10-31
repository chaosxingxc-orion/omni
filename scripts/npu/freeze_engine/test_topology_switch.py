"""Refuse changed ownership and process identity before switching topology."""
import copy
import unittest
import topology_switch as m

class IdentityTest(unittest.TestCase):
    def test_identity_changes_refused(self):
        c={'name':'saea-real500-tp4-i0','profile_id':'qwen3-omni-real500-2xtp4-s96-b2048-v1'}
        good={'Name':'/'+c['name'],'Id':'original','Image':m.common.IMAGE,'Config':{'Labels':{m.common.LABEL:c['profile_id']}},'State':{'Running':True},'RestartCount':0,'HostConfig':{'RestartPolicy':{'Name':'no'}}}
        m.validate(good,c,'original')
        for key,value in [('Id','replacement'),('RestartCount',1),('Image','foreign'),('Config',{'Labels':{}}),('State',{'Running':False})]:
            changed=copy.deepcopy(good);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate(changed,c,'original')

if __name__=='__main__':unittest.main()
