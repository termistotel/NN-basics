��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.0-beta12v2.0.0-beta0-16-g1d91213fe78��
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
}
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_class
loc:@Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
�
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2*
dtype0
�
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
_output_shapes
: *
shape: 
�
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
�
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*%
_class
loc:@Adam/learning_rate*
dtype0*
_output_shapes
: 
�
iris/dense_1/kernelVarHandleOp*$
shared_nameiris/dense_1/kernel*
dtype0*
_output_shapes
: *
shape
:d

�
'iris/dense_1/kernel/Read/ReadVariableOpReadVariableOpiris/dense_1/kernel*
_output_shapes

:d
*&
_class
loc:@iris/dense_1/kernel*
dtype0
z
iris/dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*"
shared_nameiris/dense_1/bias
�
%iris/dense_1/bias/Read/ReadVariableOpReadVariableOpiris/dense_1/bias*$
_class
loc:@iris/dense_1/bias*
dtype0*
_output_shapes
:


iris/dense/kernelVarHandleOp*
shape:	�d*"
shared_nameiris/dense/kernel*
dtype0*
_output_shapes
: 
�
%iris/dense/kernel/Read/ReadVariableOpReadVariableOpiris/dense/kernel*
_output_shapes
:	�d*$
_class
loc:@iris/dense/kernel*
dtype0
v
iris/dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:d* 
shared_nameiris/dense/bias
�
#iris/dense/bias/Read/ReadVariableOpReadVariableOpiris/dense/bias*"
_class
loc:@iris/dense/bias*
dtype0*
_output_shapes
:d
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
�
Adam/iris/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
shape
:d
*+
shared_nameAdam/iris/dense_1/kernel/m*
dtype0
�
.Adam/iris/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/iris/dense_1/kernel/m*-
_class#
!loc:@Adam/iris/dense_1/kernel/m*
dtype0*
_output_shapes

:d

�
Adam/iris/dense_1/bias/mVarHandleOp*
shape:
*)
shared_nameAdam/iris/dense_1/bias/m*
dtype0*
_output_shapes
: 
�
,Adam/iris/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/iris/dense_1/bias/m*
dtype0*
_output_shapes
:
*+
_class!
loc:@Adam/iris/dense_1/bias/m
�
Adam/iris/dense/kernel/mVarHandleOp*
shape:	�d*)
shared_nameAdam/iris/dense/kernel/m*
dtype0*
_output_shapes
: 
�
,Adam/iris/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/iris/dense/kernel/m*+
_class!
loc:@Adam/iris/dense/kernel/m*
dtype0*
_output_shapes
:	�d
�
Adam/iris/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:d*'
shared_nameAdam/iris/dense/bias/m
�
*Adam/iris/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/iris/dense/bias/m*)
_class
loc:@Adam/iris/dense/bias/m*
dtype0*
_output_shapes
:d
�
Adam/iris/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:d
*+
shared_nameAdam/iris/dense_1/kernel/v
�
.Adam/iris/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/iris/dense_1/kernel/v*
_output_shapes

:d
*-
_class#
!loc:@Adam/iris/dense_1/kernel/v*
dtype0
�
Adam/iris/dense_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*)
shared_nameAdam/iris/dense_1/bias/v
�
,Adam/iris/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/iris/dense_1/bias/v*+
_class!
loc:@Adam/iris/dense_1/bias/v*
dtype0*
_output_shapes
:

�
Adam/iris/dense/kernel/vVarHandleOp*
shape:	�d*)
shared_nameAdam/iris/dense/kernel/v*
dtype0*
_output_shapes
: 
�
,Adam/iris/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/iris/dense/kernel/v*+
_class!
loc:@Adam/iris/dense/kernel/v*
dtype0*
_output_shapes
:	�d
�
Adam/iris/dense/bias/vVarHandleOp*
shape:d*'
shared_nameAdam/iris/dense/bias/v*
dtype0*
_output_shapes
: 
�
*Adam/iris/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/iris/dense/bias/v*)
_class
loc:@Adam/iris/dense/bias/v*
dtype0*
_output_shapes
:d

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *�
value�B� B�
{
lays
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures

0
	1

2
�
iter

beta_1

beta_2
	decay
learning_ratem>m?m@mAvBvCvDvE
 

0
1
2
3

0
1
2
3
y
metrics
non_trainable_variables
regularization_losses
trainable_variables
	variables

layers
 
�

kernel
bias
_callable_losses
_eager_losses
regularization_losses
trainable_variables
	variables
	keras_api
�

kernel
bias
_callable_losses
_eager_losses
regularization_losses
 trainable_variables
!	variables
"	keras_api
{
#_callable_losses
$_eager_losses
%regularization_losses
&trainable_variables
'	variables
(	keras_api
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEiris/dense_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEiris/dense_1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEiris/dense/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEiris/dense/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

)0
 

0
	1

2
 
 
 

0
1

0
1
y
*metrics
+non_trainable_variables
regularization_losses
trainable_variables
	variables

,layers
 
 
 

0
1

0
1
y
-metrics
.non_trainable_variables
regularization_losses
 trainable_variables
!	variables

/layers
 
 
 
 
 
y
0metrics
1non_trainable_variables
%regularization_losses
&trainable_variables
'	variables

2layers
�
	3total
	4count
5
_fn_kwargs
6_updates
7regularization_losses
8trainable_variables
9	variables
:	keras_api
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

30
41
y
;metrics
<non_trainable_variables
7regularization_losses
8trainable_variables
9	variables

=layers
 

30
41
 
|z
VARIABLE_VALUEAdam/iris/dense_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/iris/dense_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/iris/dense/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/iris/dense/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/iris/dense_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/iris/dense_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/iris/dense/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/iris/dense/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
�
serving_default_input_1Placeholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1iris/dense/kerneliris/dense/biasiris/dense_1/kerneliris/dense_1/bias*
Tin	
2*'
_output_shapes
:���������
*+
f&R$
"__inference_signature_wrapper_4752*
Tout
2**
config_proto

GPU 

CPU2J 8
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'iris/dense_1/kernel/Read/ReadVariableOp%iris/dense_1/bias/Read/ReadVariableOp%iris/dense/kernel/Read/ReadVariableOp#iris/dense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/iris/dense_1/kernel/m/Read/ReadVariableOp,Adam/iris/dense_1/bias/m/Read/ReadVariableOp,Adam/iris/dense/kernel/m/Read/ReadVariableOp*Adam/iris/dense/bias/m/Read/ReadVariableOp.Adam/iris/dense_1/kernel/v/Read/ReadVariableOp,Adam/iris/dense_1/bias/v/Read/ReadVariableOp,Adam/iris/dense/kernel/v/Read/ReadVariableOp*Adam/iris/dense/bias/v/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-4836*&
f!R
__inference__traced_save_4835*
Tout
2**
config_proto

GPU 

CPU2J 8* 
Tin
2	*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateiris/dense_1/kerneliris/dense_1/biasiris/dense/kerneliris/dense/biastotalcountAdam/iris/dense_1/kernel/mAdam/iris/dense_1/bias/mAdam/iris/dense/kernel/mAdam/iris/dense/bias/mAdam/iris/dense_1/kernel/vAdam/iris/dense_1/bias/vAdam/iris/dense/kernel/vAdam/iris/dense/bias/v*+
_gradient_op_typePartitionedCall-4906*)
f$R"
 __inference__traced_restore_4905*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: ��
�
�
$__inference_dense_layer_call_fn_4640

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4635*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4629*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������d*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
#__inference_iris_layer_call_fn_4733
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������
*
Tin	
2*+
_gradient_op_typePartitionedCall-4726*G
fBR@
>__inference_iris_layer_call_and_return_conditional_losses_4725*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*:
_input_shapes)
':���������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_4657

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������
*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:���������
*
T0�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
>__inference_iris_layer_call_and_return_conditional_losses_4725

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4605*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4611�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������d*
Tin
2*+
_gradient_op_typePartitionedCall-4635*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4629*
Tout
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4657*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������
*+
_gradient_op_typePartitionedCall-4663�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:���������
*
T0"
identityIdentity:output:0*:
_input_shapes)
':���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
�
�
&__inference_dense_1_layer_call_fn_4668

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������
*+
_gradient_op_typePartitionedCall-4663*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4657*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_4629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�di
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������d*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
�
�
__inference__wrapped_model_4587
input_1-
)iris_dense_matmul_readvariableop_resource.
*iris_dense_biasadd_readvariableop_resource/
+iris_dense_1_matmul_readvariableop_resource0
,iris_dense_1_biasadd_readvariableop_resource
identity��!iris/dense/BiasAdd/ReadVariableOp� iris/dense/MatMul/ReadVariableOp�#iris/dense_1/BiasAdd/ReadVariableOp�"iris/dense_1/MatMul/ReadVariableOpI
iris/flatten/ShapeShapeinput_1*
T0*
_output_shapes
:j
 iris/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:l
"iris/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:l
"iris/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
iris/flatten/strided_sliceStridedSliceiris/flatten/Shape:output:0)iris/flatten/strided_slice/stack:output:0+iris/flatten/strided_slice/stack_1:output:0+iris/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: g
iris/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
����������
iris/flatten/Reshape/shapePack#iris/flatten/strided_slice:output:0%iris/flatten/Reshape/shape/1:output:0*
_output_shapes
:*
T0*
N�
iris/flatten/ReshapeReshapeinput_1#iris/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
 iris/dense/MatMul/ReadVariableOpReadVariableOp)iris_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�d�
iris/dense/MatMulMatMuliris/flatten/Reshape:output:0(iris/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������d*
T0�
!iris/dense/BiasAdd/ReadVariableOpReadVariableOp*iris_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d�
iris/dense/BiasAddBiasAddiris/dense/MatMul:product:0)iris/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������df
iris/dense/ReluReluiris/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
"iris/dense_1/MatMul/ReadVariableOpReadVariableOp+iris_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d
�
iris/dense_1/MatMulMatMuliris/dense/Relu:activations:0*iris/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
#iris/dense_1/BiasAdd/ReadVariableOpReadVariableOp,iris_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0�
iris/dense_1/BiasAddBiasAddiris/dense_1/MatMul:product:0+iris/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
p
iris/dense_1/SoftmaxSoftmaxiris/dense_1/BiasAdd:output:0*'
_output_shapes
:���������
*
T0�
IdentityIdentityiris/dense_1/Softmax:softmax:0"^iris/dense/BiasAdd/ReadVariableOp!^iris/dense/MatMul/ReadVariableOp$^iris/dense_1/BiasAdd/ReadVariableOp#^iris/dense_1/MatMul/ReadVariableOp*'
_output_shapes
:���������
*
T0"
identityIdentity:output:0*:
_input_shapes)
':���������::::2H
"iris/dense_1/MatMul/ReadVariableOp"iris/dense_1/MatMul/ReadVariableOp2J
#iris/dense_1/BiasAdd/ReadVariableOp#iris/dense_1/BiasAdd/ReadVariableOp2D
 iris/dense/MatMul/ReadVariableOp iris/dense/MatMul/ReadVariableOp2F
!iris/dense/BiasAdd/ReadVariableOp!iris/dense/BiasAdd/ReadVariableOp:' #
!
_user_specified_name	input_1: : : : 
�
�
>__inference_iris_layer_call_and_return_conditional_losses_4702

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4611*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4605�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������d*+
_gradient_op_typePartitionedCall-4635*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4629*
Tout
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������
*
Tin
2*+
_gradient_op_typePartitionedCall-4663*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4657�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*:
_input_shapes)
':���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
�-
�
__inference__traced_save_4835
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_iris_dense_1_kernel_read_readvariableop0
,savev2_iris_dense_1_bias_read_readvariableop0
,savev2_iris_dense_kernel_read_readvariableop.
*savev2_iris_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_iris_dense_1_kernel_m_read_readvariableop7
3savev2_adam_iris_dense_1_bias_m_read_readvariableop7
3savev2_adam_iris_dense_kernel_m_read_readvariableop5
1savev2_adam_iris_dense_bias_m_read_readvariableop9
5savev2_adam_iris_dense_1_kernel_v_read_readvariableop7
3savev2_adam_iris_dense_1_bias_v_read_readvariableop7
3savev2_adam_iris_dense_kernel_v_read_readvariableop5
1savev2_adam_iris_dense_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_a5f20af2e59d42d791fc661eafe97f07/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*�	
value�	B�	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_iris_dense_1_kernel_read_readvariableop,savev2_iris_dense_1_bias_read_readvariableop,savev2_iris_dense_kernel_read_readvariableop*savev2_iris_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_iris_dense_1_kernel_m_read_readvariableop3savev2_adam_iris_dense_1_bias_m_read_readvariableop3savev2_adam_iris_dense_kernel_m_read_readvariableop1savev2_adam_iris_dense_bias_m_read_readvariableop5savev2_adam_iris_dense_1_kernel_v_read_readvariableop3savev2_adam_iris_dense_1_bias_v_read_readvariableop3savev2_adam_iris_dense_kernel_v_read_readvariableop1savev2_adam_iris_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapesw
u: : : : : : :d
:
:	�d:d: : :d
:
:	�d:d:d
:
:	�d:d: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
�	
]
A__inference_flatten_layer_call_and_return_conditional_losses_4605

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0Z
Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:����������*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:����������*
T0"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_4752
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*'
_output_shapes
:���������
*+
_gradient_op_typePartitionedCall-4745*(
f#R!
__inference__wrapped_model_4587*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*:
_input_shapes)
':���������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�
�
#__inference_iris_layer_call_fn_4710
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:���������
*+
_gradient_op_typePartitionedCall-4703*G
fBR@
>__inference_iris_layer_call_and_return_conditional_losses_4702*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������
*
T0"
identityIdentity:output:0*:
_input_shapes)
':���������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�
�
>__inference_iris_layer_call_and_return_conditional_losses_4675
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinput_1*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4605*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4611�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4635*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4629*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������d*
Tin
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������
*+
_gradient_op_typePartitionedCall-4663*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4657*
Tout
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*:
_input_shapes)
':���������::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�
�
>__inference_iris_layer_call_and_return_conditional_losses_4688
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinput_1*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4605*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4611�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4629*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������d*
Tin
2*+
_gradient_op_typePartitionedCall-4635�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4663*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4657*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������
�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*:
_input_shapes)
':���������::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�
B
&__inference_flatten_layer_call_fn_4614

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-4611*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4605*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:����������*
T0"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�L
�

 __inference__traced_restore_4905
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate*
&assignvariableop_5_iris_dense_1_kernel(
$assignvariableop_6_iris_dense_1_bias(
$assignvariableop_7_iris_dense_kernel&
"assignvariableop_8_iris_dense_bias
assignvariableop_9_total
assignvariableop_10_count2
.assignvariableop_11_adam_iris_dense_1_kernel_m0
,assignvariableop_12_adam_iris_dense_1_bias_m0
,assignvariableop_13_adam_iris_dense_kernel_m.
*assignvariableop_14_adam_iris_dense_bias_m2
.assignvariableop_15_adam_iris_dense_1_kernel_v0
,assignvariableop_16_adam_iris_dense_1_bias_v0
,assignvariableop_17_adam_iris_dense_kernel_v.
*assignvariableop_18_adam_iris_dense_bias_v
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�	
RestoreV2/tensor_namesConst"/device:CPU:0*�	
value�	B�	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*9
value0B.B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*!
dtypes
2	*`
_output_shapesN
L:::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:v
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:~
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0}
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_iris_dense_1_kernelIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_iris_dense_1_biasIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_iris_dense_kernelIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_iris_dense_biasIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0{
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_adam_iris_dense_1_kernel_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp,assignvariableop_12_adam_iris_dense_1_bias_mIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_iris_dense_kernel_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_iris_dense_bias_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_iris_dense_1_kernel_vIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_iris_dense_1_bias_vIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_iris_dense_kernel_vIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_iris_dense_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype0�
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : "7L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�_
�
lays
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
F__call__
G_default_save_signature
*H&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "Iris", "name": "iris", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Iris"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
5
0
	1

2"
trackable_list_wrapper
�
iter

beta_1

beta_2
	decay
learning_ratem>m?m@mAvBvCvDvE"
	optimizer
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
metrics
non_trainable_variables
regularization_losses
trainable_variables
	variables

layers
F__call__
G_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
�

kernel
bias
_callable_losses
_eager_losses
regularization_losses
trainable_variables
	variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "activity_regularizer": null}
�

kernel
bias
_callable_losses
_eager_losses
regularization_losses
 trainable_variables
!	variables
"	keras_api
L__call__
*M&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "activity_regularizer": null}
�
#_callable_losses
$_eager_losses
%regularization_losses
&trainable_variables
'	variables
(	keras_api
N__call__
*O&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "activity_regularizer": null}
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#d
2iris/dense_1/kernel
:
2iris/dense_1/bias
$:"	�d2iris/dense/kernel
:d2iris/dense/bias
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
*metrics
+non_trainable_variables
regularization_losses
trainable_variables
	variables

,layers
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
-metrics
.non_trainable_variables
regularization_losses
 trainable_variables
!	variables

/layers
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0metrics
1non_trainable_variables
%regularization_losses
&trainable_variables
'	variables

2layers
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
	3total
	4count
5
_fn_kwargs
6_updates
7regularization_losses
8trainable_variables
9	variables
:	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
;metrics
<non_trainable_variables
7regularization_losses
8trainable_variables
9	variables

=layers
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
*:(d
2Adam/iris/dense_1/kernel/m
$:"
2Adam/iris/dense_1/bias/m
):'	�d2Adam/iris/dense/kernel/m
": d2Adam/iris/dense/bias/m
*:(d
2Adam/iris/dense_1/kernel/v
$:"
2Adam/iris/dense_1/bias/v
):'	�d2Adam/iris/dense/kernel/v
": d2Adam/iris/dense/bias/v
�2�
#__inference_iris_layer_call_fn_4710
#__inference_iris_layer_call_fn_4733�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference__wrapped_model_4587�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
>__inference_iris_layer_call_and_return_conditional_losses_4675
>__inference_iris_layer_call_and_return_conditional_losses_4725
>__inference_iris_layer_call_and_return_conditional_losses_4688
>__inference_iris_layer_call_and_return_conditional_losses_4702�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
1B/
"__inference_signature_wrapper_4752input_1
�2�
&__inference_dense_1_layer_call_fn_4668�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_4657�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_4640�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_4629�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_flatten_layer_call_fn_4614�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_4605�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference_iris_layer_call_fn_4733Z8�5
.�+
%�"
input_1���������
p
� "����������
�
>__inference_iris_layer_call_and_return_conditional_losses_4702f7�4
-�*
$�!
inputs���������
p 
� "%�"
�
0���������

� �
?__inference_dense_layer_call_and_return_conditional_losses_4629]0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� �
A__inference_dense_1_layer_call_and_return_conditional_losses_4657\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������

� z
&__inference_flatten_layer_call_fn_4614P3�0
)�&
$�!
inputs���������
� "������������
>__inference_iris_layer_call_and_return_conditional_losses_4675g8�5
.�+
%�"
input_1���������
p 
� "%�"
�
0���������

� �
>__inference_iris_layer_call_and_return_conditional_losses_4725f7�4
-�*
$�!
inputs���������
p
� "%�"
�
0���������

� y
&__inference_dense_1_layer_call_fn_4668O/�,
%�"
 �
inputs���������d
� "����������
�
__inference__wrapped_model_4587q4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1���������
�
#__inference_iris_layer_call_fn_4710Z8�5
.�+
%�"
input_1���������
p 
� "����������
�
>__inference_iris_layer_call_and_return_conditional_losses_4688g8�5
.�+
%�"
input_1���������
p
� "%�"
�
0���������

� �
"__inference_signature_wrapper_4752|?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������
x
$__inference_dense_layer_call_fn_4640P0�-
&�#
!�
inputs����������
� "����������d�
A__inference_flatten_layer_call_and_return_conditional_losses_4605]3�0
)�&
$�!
inputs���������
� "&�#
�
0����������
� 