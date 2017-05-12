import unittest

from src.models import BratsModels


class TestBratsModels(unittest.TestCase):
    def setUp(self):
        # Set up configuration variables
        self.num_classes = 5
        self.num_modalities = 4

    def test_onepathway(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.one_pathway(self.num_modalities, (25, 25, 25), self.num_classes),
                        msg='One pathway model does not compile')
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (25, 25, 25), self.num_classes,
                                              model_name='one_pathway'),
                        msg='One pathway model cannot be retrieved using the BratsModels class method get()')

    def test_onepathway_no_downsampling(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.one_pathway_no_downsampling(
            self.num_modalities, (32, 32, 32),
            self.num_classes),
            msg='One pathway without downsampling model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (32, 32, 32), self.num_classes,
                                              model_name='one_pathway_no_downsampling'),
                        msg='One pathway without downsampling model cannot be '
                            'retrieved using the BratsModels class method get()')

    def test_unet(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.u_net(
            self.num_modalities, (32, 32, 32),
            self.num_classes),
            msg='U-net model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (32, 32, 32), self.num_classes,
                                              model_name='u_net'),
                        msg='U-net model cannot be retrieved using the BratsModels class method get()')

    def test_onepathway_upsampling(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.one_pathway_skipped_upsampling(
            self.num_modalities, (32, 32, 32),
            self.num_classes),
            msg='One pathway skipped upsampling model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (32, 32, 32), self.num_classes,
                                              model_name='one_pathway_skipped_upsampling'),
                        msg='One pathway skipped upsampling model cannot be '
                            'retrieved using the BratsModels class method get()')

    def test_fcn_8(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.fcn_8(
            self.num_modalities, (32, 32, 32),
            self.num_classes),
            msg='FCN 8 (Long) model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (32, 32, 32), self.num_classes,
                                              model_name='fcn_8'),
                        msg='FCN 8 (Long) model cannot be '
                            'retrieved using the BratsModels class method get()')

    def test_two_pathways(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.two_pathways(
            self.num_modalities, (32, 32, 32),
            self.num_classes),
            msg='Two pathways model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (32, 32, 32), self.num_classes,
                                              model_name='two_pathways'),
                        msg='Two pathways model cannot be '
                            'retrieved using the BratsModels class method get()')
        model, dummy = BratsModels.get_model(self.num_modalities, (64, 64, 64), self.num_classes,
                                             model_name='fcn_8')

        model.summary()

    def test_deepmedic(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.deepmedic(
            self.num_modalities, (57,57,57),
            self.num_classes,segment_dimensions_up=(25,25,25)),
            msg='Two pathways model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(BratsModels.get_model(self.num_modalities, (57,57,57), self.num_classes, segment_dimensions_up=(25,25,25),
                                              model_name='deepmedic'),
                        msg='Two pathways model cannot be '
                            'retrieved using the BratsModels class method get()')

        model, dummy = BratsModels.get_model(self.num_modalities, (57,57,57), self.num_classes,
                                             model_name='deepmedic',segment_dimensions_up=(25,25,25))

        model.summary()


    def test_two_pathwyas_dense(self):
        # Try to get model using the static method and see if it fails to compile
        self.assertTrue(BratsModels.deepmedic(
            self.num_modalities, (57, 57, 57),
            self.num_classes, segment_dimensions_up=(25, 25, 25)),
            msg='Two pathways model does not compile'
        )
        # Try to get model using the get method
        self.assertTrue(
            BratsModels.get_model(self.num_modalities, (57, 57, 57), self.num_classes, segment_dimensions_up=(25, 25, 25),
                                  model_name='deepmedic'),
            msg='Two pathways model cannot be '
                'retrieved using the BratsModels class method get()')

        model, dummy = BratsModels.get_model(self.num_modalities, (57, 57, 57), self.num_classes,
                                             model_name='deepmedic', segment_dimensions_up=(25, 25, 25))

        model.summary()


if __name__ == '__main__':
    unittest.main()
