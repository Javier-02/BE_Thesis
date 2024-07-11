import logging
import os
from typing import Annotated, Optional

import vtk
import itk

import numpy as np

import slicer

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

#
# ElasticRegistration
#

class ElasticRegistration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ElasticRegistration")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#ElasticRegistration">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")


#
# ElasticRegistrationParameterNode
#

@parameterNodeWrapper
class ElasticRegistrationParameterNode:
    """
    The parameters needed by module.

    syntheticCT - The first volume to register.
    planificationCT - The second volume to register.
    outputVolume - The output volume that will contain the registration result.
    """

    syntheticCT: vtkMRMLScalarVolumeNode
    planificationCT: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode

#
# ElasticRegistrationWidget
#

class ElasticRegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ElasticRegistration.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ElasticRegistrationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input volume nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.syntheticCT:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.syntheticCT = firstVolumeNode
        if not self._parameterNode.planificationCT:
            secondVolumeNode = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLScalarVolumeNode")
            if secondVolumeNode:
                self._parameterNode.planificationCT = secondVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[ElasticRegistrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

        # Initial GUI update
        self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if not self._parameterNode:
            self.ui.applyButton.enabled = False
            return

        # Check if all required parameters are set
        isReady = (
            self._parameterNode.syntheticCT
            and self._parameterNode.planificationCT
            and self._parameterNode.outputVolume
        )
        self.ui.applyButton.enabled = bool(isReady)

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            self.logic.process(self._parameterNode)
        except Exception as e:
            slicer.util.errorDisplay(_("Failed to compute results: {0}").format(str(e)))
            import traceback

            traceback.print_exc()

#
# ElasticRegistrationLogic
#

class ElasticRegistrationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module. The interface should be such that
    other python code can import this class and make use of the
    functionality without requiring an instance of the Widget
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode: ElasticRegistrationParameterNode) -> None:
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.syntheticCT:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            parameterNode.syntheticCT = firstVolumeNode if firstVolumeNode else None
        if not parameterNode.planificationCT:
            secondVolumeNode = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLScalarVolumeNode")
            parameterNode.planificationCT = secondVolumeNode if secondVolumeNode else None
        if not parameterNode.outputVolume:
            parameterNode.outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "OutputVolume")

    def getParameterNode(self):
        return ElasticRegistrationParameterNode(super().getParameterNode())
    
    def fuse_images(self, result_image_bspline, inverse_rigid_hu):
                """Fuse two images based on specific conditions."""
                result_bspline_array = itk.GetArrayViewFromImage(result_image_bspline)
                inverse_rigid_array = itk.GetArrayViewFromImage(inverse_rigid_hu)

                # Ensure the arrays have the same shape
                if result_bspline_array.shape != inverse_rigid_array.shape:
                    raise ValueError("Images to be fused must have the same dimensions")

                # Iterate through slices and fuse where conditions are met
                for i in range(result_bspline_array.shape[0]):
                    # Condition 1: Check if the slice in inverse_rigid_hu is a specific value
                    if np.all(inverse_rigid_array[i, :, :] == np.min(inverse_rigid_array[0, :, :])):
                        inverse_rigid_array[i, :, :] = result_bspline_array[i, :, :]

                    # Condition 2: Check if the slice in inverse_rigid_hu is another specific value
                    if np.all(inverse_rigid_array[i, :, :] == np.min(inverse_rigid_array[-1, :, :])):
                        inverse_rigid_array[i, :, :] = result_bspline_array[i, :, :]

                # Convert back into ITK format
                fused_image = itk.GetImageFromArray(inverse_rigid_array)
                fused_image.SetSpacing(inverse_rigid_hu.GetSpacing())
                fused_image.SetOrigin(inverse_rigid_hu.GetOrigin())
                fused_image.SetDirection(inverse_rigid_hu.GetDirection())
                return fused_image # fused itk image

    def process(self, parameterNode: ElasticRegistrationParameterNode) -> None:
        """
        Run the processing algorithm.
        """
        if not parameterNode.syntheticCT or not parameterNode.planificationCT or not parameterNode.outputVolume:
            raise ValueError("Input or output volume is invalid")

        logging.info("Processing started")

        # Get input volumes metadata
        syntheticCTNode = parameterNode.syntheticCT
        dirs = vtk.vtkMatrix4x4()

        origin = syntheticCTNode.GetOrigin()
        spacing = syntheticCTNode.GetSpacing()
        syntheticCTNode.GetIJKToRASDirectionMatrix(dirs)

        # Get the numpy arrays from the vtkMRMLScalarVolumeNode
        syntheticCTImageData = slicer.util.arrayFromVolume(parameterNode.syntheticCT)
        planificationCTImageData = slicer.util.arrayFromVolume(parameterNode.planificationCT)

        # Get itk images from numpy arrays
        syntheticCTImage = itk.image_view_from_array(syntheticCTImageData)
        planificationCTImage = itk.image_view_from_array(planificationCTImageData)

        # HU correction
        hu_min = -1024
        hu_max = 3072

        # Get array from ITK to perform operations
        syntheticCTArray = itk.GetArrayViewFromImage(syntheticCTImage)
        planificationCTArray = itk.GetArrayViewFromImage(planificationCTImage)

        # Applying inverse of min-max scaling (x' = (x-xmin)/(xmax-xmin))
        if np.min(planificationCTArray) >= 0 and np.max(planificationCTArray) <= 1:
            planificationCTArray = np.round(planificationCTArray * (hu_max - hu_min) + hu_min)
        else:
            planificationCTArray = planificationCTArray

        if np.min(syntheticCTArray) >= 0 and np.max(syntheticCTArray) <= 1:
            syntheticCTArray = syntheticCTArray * (hu_max - hu_min) + hu_min
            syntheticCTArray[syntheticCTArray == -801.754630] = -999
            syntheticCTArray = np.round(syntheticCTArray)
        else:
            syntheticCTArray = syntheticCTArray

        # Convert back into ITK format
        itkPlanificationCT = itk.image_view_from_array(planificationCTArray)
        itkSyntheticCT = itk.image_view_from_array(syntheticCTArray)

        print('HU CORRECTION DONE')

        # Proceed with the rigid registration considering syntheticCT as moving and planificationCT as fixed
        # Rigid Registration
        parameter_object_rigid = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object_rigid.GetDefaultParameterMap('rigid')
        default_rigid_parameter_map['AutomaticTransformInitialization'] = ['true']
        default_rigid_parameter_map['Interpolator'] = ['NearestNeighborInterpolator']
        parameter_object_rigid.AddParameterMap(default_rigid_parameter_map)

        # Call registration function
        inverse_rigid, rigid_transform_parameters = itk.elastix_registration_method(
            itkPlanificationCT, # fixed image
            itkSyntheticCT, # moving image
            parameter_object=parameter_object_rigid,
            log_to_console=False)

        # Get array from ITK to perform operations
        inverse_rigid_array = itk.GetArrayViewFromImage(inverse_rigid)
        # print('Minimum pixel value in inverse rigid image is:', np.min(inverse_rigid_array))

        # Assign the median to the zero elements 
        inverse_rigid_array[inverse_rigid_array == 0] = np.min(inverse_rigid_array)

        # Convert back into ITK format
        inverse_rigid_hu = itk.image_view_from_array(inverse_rigid_array)

        print('RIGID REGISTRATION DONE')

        # Create an output volume node
        outputVolumeNode = parameterNode.outputVolume
        slicer.util.updateVolumeFromArray(outputVolumeNode, inverse_rigid_hu)

        # Set the correct spacing, origin, and direction to the output volume
        outputVolumeNode.SetSpacing(spacing)
        outputVolumeNode.SetOrigin(origin)
        outputVolumeNode.SetIJKToRASDirectionMatrix(dirs)

        # Display the result
        slicer.util.setSliceViewerLayers(background=parameterNode.planificationCT, foreground=outputVolumeNode, foregroundOpacity=1)

        logging.info("Processing completed")

        # Affine Registration
        parameter_object_affine = itk.ParameterObject.New()
        parameter_map_affine = parameter_object_affine.GetDefaultParameterMap("affine")
        parameter_map_affine["Metric"] = ["AdvancedMattesMutualInformation"]
        parameter_map_affine['AutomaticTransformInitialization'] = ['true']
        parameter_map_affine['Interpolator'] = ['NearestNeighborInterpolator']
        parameter_object_affine.AddParameterMap(parameter_map_affine)

        # Call affine registration function
        affine_image, affine_transform_parameters = itk.elastix_registration_method(
            inverse_rigid_hu, # fixed image
            itkSyntheticCT, # moving image
            parameter_object=parameter_object_affine,
            log_to_console=False)
        
        print('AFFINE REGISTRATION DONE')

        # B-spline Registration
        parameter_object_bspline = itk.ParameterObject.New()
        default_bspline_parameter_map = parameter_object_bspline.GetDefaultParameterMap('bspline', 1)
        default_bspline_parameter_map['FinalBSplineInterpolationOrder'] = ['3']
        default_bspline_parameter_map['NumberOfResolutions '] = ['4']
        default_bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['16']
        parameter_object_bspline.AddParameterMap(default_bspline_parameter_map)

        # Call registration function
        result_image_bspline, result_transform_parameters = itk.elastix_registration_method(
            inverse_rigid_hu, # fixed image
            affine_image, # moving image
            parameter_object=parameter_object_bspline,
            log_to_console=True)
        
        print('BSPLINE REGISTRATION DONE')
        
        # Fuse Images
        fused_image = self.fuse_images(result_image_bspline, inverse_rigid_hu)
        # Set fused_image as the output volume
        # Get the output image and convert it back to a numpy array
        fused_image_arr = itk.GetArrayViewFromImage(fused_image)

        print('FUSION DONE')
        print('REGISTRATION COMPLETED')

