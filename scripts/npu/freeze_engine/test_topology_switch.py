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

class RecoveryTest(unittest.TestCase):
    def test_only_exact_stopped_owned_archive_is_accepted(self):
        c={'name':'old','profile_id':'owned'}
        good={'Name':'/archive','Id':'original','Image':m.common.IMAGE,
              'Config':{'Labels':{m.common.LABEL:'owned'}},'State':{'Running':False},'RestartCount':0}
        m.validate_stopped_archive(good,c,'original','archive')
        for key,value in [('Id','new'),('Name','/wrong'),('Image','foreign'),
                          ('Config',{'Labels':{}}),('State',{'Running':True}),('RestartCount',1)]:
            changed=copy.deepcopy(good);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.validate_stopped_archive(changed,c,'original','archive')

if __name__=='__main__':unittest.main()
