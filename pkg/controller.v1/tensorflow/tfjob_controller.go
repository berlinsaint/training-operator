// Copyright 2021 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

import (
	"context"
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/cache"
	"reflect"
	"sort"

	"github.com/kubeflow/common/pkg/util/k8sutil"
	"k8s.io/apimachinery/pkg/util/rand"
	"strconv"
	"strings"
	"time"
	"volcano.sh/apis/pkg/apis/scheduling/v1beta1"

	"github.com/go-logr/logr"
	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	"github.com/kubeflow/common/pkg/controller.v1/common"
	"github.com/kubeflow/common/pkg/controller.v1/control"
	"github.com/kubeflow/common/pkg/controller.v1/expectation"
	commonutil "github.com/kubeflow/common/pkg/util"
	train_util "github.com/kubeflow/common/pkg/util/train"
	tensorflowv1 "github.com/kubeflow/training-operator/pkg/apis/tensorflow/v1"
	tfv1 "github.com/kubeflow/training-operator/pkg/apis/tensorflow/v1"
	"github.com/kubeflow/training-operator/pkg/apis/tensorflow/validation"
	trainingoperatorcommon "github.com/kubeflow/training-operator/pkg/common"
	"github.com/kubeflow/training-operator/pkg/common/util"
	"github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/source"
	volcanoclient "volcano.sh/apis/pkg/client/clientset/versioned"
)

var (
	KeyFunc                       = cache.DeletionHandlingMetaNamespaceKeyFunc
	succeededServiceCreationCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "succeeded_tfjob_service_creation_total",
		Help: "The total number of succeeded service creation",
	})
	failedServiceCreationCount = promauto.NewCounter(prometheus.CounterOpts{
		Name: "failed_tfjob_service_creation_total",
		Help: "The total number of failed service creation",
	})
)

const (

	// tfJobSucceededReason is added in a tfjob when it is succeeded.
	tfJobSucceededReason = "TFJobSucceeded"
	// tfJobRunningReason is added in a tfjob when it is running.
	tfJobRunningReason = "TFJobRunning"
	// tfJobFailedReason is added in a tfjob when it is failed.
	tfJobFailedReason = "TFJobFailed"
	// tfJobRestarting is added in a tfjob when it is restarting.
	tfJobRestartingReason = "TFJobRestarting"

	FailedDeleteJobReason     = "FailedDeleteJob"
	SuccessfulDeleteJobReason = "SuccessfulDeleteJob"

	controllerName = "tfjob-controller"

	// labels for pods and servers.
	tfReplicaTypeLabel  = "replica-type"
	tfReplicaIndexLabel = "replica-index"
	// volcanoTaskSpecKey task spec key used in pod annotation when EnableGangScheduling is true
	volcanoTaskSpecKey = "volcano.sh/task-spec"

	// gang scheduler name.
	gangSchedulerName = "volcano"
	// tfConfig is the environment variable name of TensorFlow cluster spec.
	tfConfig = "TF_CONFIG"
	// exitedWithCodeReason is the normal reason when the pod is exited because of the exit code.
	exitedWithCodeReason = "ExitedWithCode"
	// podTemplateRestartPolicyReason is the warning reason when the restart
	// policy is set in pod template.
	podTemplateRestartPolicyReason = "SettedPodTemplateRestartPolicy"
	// podTemplateSchedulerNameReason is the warning reason when other scheduler name is set
	// in pod templates with gang-scheduling enabled
	podTemplateSchedulerNameReason = "SettedPodTemplateSchedulerName"
	// gangSchedulingPodGroupAnnotation is the annotation key used by batch schedulers
	gangSchedulingPodGroupAnnotation = "scheduling.k8s.io/group-name"
)

func NewReconciler(mgr manager.Manager, enableGangScheduling bool) *TFJobReconciler {
	r := &TFJobReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		recorder: mgr.GetEventRecorderFor(controllerName),
		Log:      log.Log,
	}

	cfg := mgr.GetConfig()
	kubeClientSet := kubeclientset.NewForConfigOrDie(cfg)
	volcanoClientSet := volcanoclient.NewForConfigOrDie(cfg)
	sharedInformers := informers.NewSharedInformerFactory(kubeClientSet, 0)
	priorityClassInformer := sharedInformers.Scheduling().V1beta1().PriorityClasses()

	r.JobController = common.JobController{
		Controller:                  r,
		Expectations:                expectation.NewControllerExpectations(),
		Config:                      common.JobControllerConfiguration{EnableGangScheduling: enableGangScheduling},
		WorkQueue:                   &util.FakeWorkQueue{},
		Recorder:                    r.recorder,
		KubeClientSet:               kubeClientSet,
		VolcanoClientSet:            volcanoClientSet,
		PriorityClassLister:         priorityClassInformer.Lister(),
		PriorityClassInformerSynced: priorityClassInformer.Informer().HasSynced,
		PodControl:                  control.RealPodControl{KubeClient: kubeClientSet, Recorder: r.recorder},
		ServiceControl:              control.RealServiceControl{KubeClient: kubeClientSet, Recorder: r.recorder},
	}

	return r
}

// TFJobReconciler reconciles a TFJob object
type TFJobReconciler struct {
	common.JobController
	client.Client
	Scheme   *runtime.Scheme
	recorder record.EventRecorder
	Log      logr.Logger
}

// recordAbnormalPods records the active pod whose latest condition is not in True status.
func (r *TFJobReconciler) recordAbnormalPods(activePods []*v1.Pod, object runtime.Object) {
	for _, pod := range activePods {
		// If the pod starts running, should checks the container statuses rather than the conditions.
		recordContainerStatus := func(status *v1.ContainerStatus) {
			if status.State.Terminated != nil && status.State.Terminated.ExitCode != 0 {
				terminated := status.State.Terminated
				r.Recorder.Eventf(object, v1.EventTypeWarning, terminated.Reason,
					"Error pod %s container %s exitCode: %d terminated message: %s",
					pod.Name, status.Name, terminated.ExitCode, terminated.Message)
			}
			// The terminated state and waiting state don't simultaneously exists, checks them at the same time.
			if status.State.Waiting != nil && status.State.Waiting.Message != "" {
				wait := status.State.Waiting
				r.Recorder.Eventf(object, v1.EventTypeWarning, wait.Reason,
					"Error pod %s container %s waiting message: %s", pod.Name, status.Name, wait.Message)
			}
		}
		if len(pod.Status.ContainerStatuses) != 0 {
			for _, status := range pod.Status.ContainerStatuses {
				recordContainerStatus(&status)
			}
			// If the pod has container status info, that means the init container statuses are normal.
			continue
		}
		if len(pod.Status.InitContainerStatuses) != 0 {
			for _, status := range pod.Status.InitContainerStatuses {
				recordContainerStatus(&status)
			}
			continue
		}
		if len(pod.Status.Conditions) == 0 {
			continue
		}
		// Should not modify the original pod which is stored in the informer cache.
		status := pod.Status.DeepCopy()
		sort.Slice(status.Conditions, func(i, j int) bool {
			return status.Conditions[i].LastTransitionTime.After(status.Conditions[j].LastTransitionTime.Time)
		})
		condition := status.Conditions[0]
		if condition.Status == v1.ConditionTrue {
			continue
		}
		r.Recorder.Eventf(object, v1.EventTypeWarning, condition.Reason, "Error pod %s condition message: %s", pod.Name, condition.Message)
	}
}

//+kubebuilder:rbac:groups=kubeflow.org,resources=tfjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=kubeflow.org,resources=tfjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=kubeflow.org,resources=tfjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;delete

// ReconcileJobs checks and updates replicas for each given ReplicaSpec.
// It will requeue the job in case of an error while creating/deleting pods/services.
func (r *TFJobReconciler) ReconcileJobs(
	job interface{},
	replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec,
	jobStatus commonv1.JobStatus,
	runPolicy *commonv1.RunPolicy) error {

	metaObject, ok := job.(metav1.Object)
	jobName := metaObject.GetName()
	if !ok {
		return fmt.Errorf("job is not of type metav1.Object")
	}
	runtimeObject, ok := job.(runtime.Object)
	if !ok {
		return fmt.Errorf("job is not of type runtime.Object")
	}
	jobKey, err := KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for job object %#v: %v", job, err))
		return err
	}
	// Reset expectations
	// 1. Since `ReconcileJobs` is called, we expect that previous expectations are all satisfied,
	//    and it's safe to reset the expectations
	// 2. Reset expectations can avoid dirty data such as `expectedDeletion = -1`
	//    (pod or service was deleted unexpectedly)
	r.ResetExpectations(jobKey, replicas)

	logrus.Infof("Reconciling for job %s", metaObject.GetName())
	pods, err := r.Controller.GetPodsForJob(job)
	if err != nil {
		logrus.Warnf("GetPodsForJob error %v", err)
		return err
	}

	services, err := r.Controller.GetServicesForJob(job)
	if err != nil {
		logrus.Warnf("GetServicesForJob error %v", err)
		return err
	}

	oldStatus := jobStatus.DeepCopy()
	if commonutil.IsSucceeded(jobStatus) || commonutil.IsFailed(jobStatus) {
		// If the Job is succeed or failed, delete all pods and services.
		if err := r.DeletePodsAndServices(runPolicy, job, pods); err != nil {
			return err
		}

		if err := r.CleanupJob(runPolicy, jobStatus, job); err != nil {
			return err
		}

		if r.Config.EnableGangScheduling {
			r.Recorder.Event(runtimeObject, v1.EventTypeNormal, "JobTerminated", "Job has been terminated. Deleting PodGroup")
			if err := r.DeletePodGroup(metaObject); err != nil {
				r.Recorder.Eventf(runtimeObject, v1.EventTypeWarning, "FailedDeletePodGroup", "Error deleting: %v", err)
				return err
			} else {
				r.Recorder.Eventf(runtimeObject, v1.EventTypeNormal, "SuccessfulDeletePodGroup", "Deleted PodGroup: %v", jobName)
			}
		}

		// At this point the pods may have been deleted.
		// 1) If the job succeeded, we manually set the replica status.
		// 2) If any replicas are still active, set their status to succeeded.
		if commonutil.IsSucceeded(jobStatus) {
			for rtype := range jobStatus.ReplicaStatuses {
				jobStatus.ReplicaStatuses[rtype].Succeeded += jobStatus.ReplicaStatuses[rtype].Active
				jobStatus.ReplicaStatuses[rtype].Active = 0
			}
		}

		// No need to update the job status if the status hasn't changed since last time.
		if !reflect.DeepEqual(*oldStatus, jobStatus) {
			return r.Controller.UpdateJobStatusInApiServer(job, &jobStatus)
		}

		return nil
	}

	// retrieve the previous number of retry
	previousRetry := r.WorkQueue.NumRequeues(jobKey)

	activePods := k8sutil.FilterActivePods(pods)

	r.recordAbnormalPods(activePods, runtimeObject)

	active := int32(len(activePods))
	failed := k8sutil.FilterPodCount(pods, v1.PodFailed)
	totalReplicas := k8sutil.GetTotalReplicas(replicas)
	prevReplicasFailedNum := k8sutil.GetTotalFailedReplicas(jobStatus.ReplicaStatuses)

	var failureMessage string
	jobExceedsLimit := false
	exceedsBackoffLimit := false
	pastBackoffLimit := false

	if runPolicy.BackoffLimit != nil {
		jobHasNewFailure := failed > prevReplicasFailedNum
		// new failures happen when status does not reflect the failures and active
		// is different than parallelism, otherwise the previous controller loop
		// failed updating status so even if we pick up failure it is not a new one
		exceedsBackoffLimit = jobHasNewFailure && (active != totalReplicas) &&
			(int32(previousRetry)+1 > *runPolicy.BackoffLimit)

		pastBackoffLimit, err = r.PastBackoffLimit(jobName, runPolicy, replicas, pods)
		if err != nil {
			return err
		}
	}

	if exceedsBackoffLimit || pastBackoffLimit {
		// check if the number of pod restart exceeds backoff (for restart OnFailure only)
		// OR if the number of failed jobs increased since the last syncJob
		jobExceedsLimit = true
		failureMessage = fmt.Sprintf("Job %s has failed because it has reached the specified backoff limit", jobName)
	} else if r.PastActiveDeadline(runPolicy, jobStatus) {
		failureMessage = fmt.Sprintf("Job %s has failed because it was active longer than specified deadline", jobName)
		jobExceedsLimit = true
	}

	if jobExceedsLimit {
		// Set job completion time before resource cleanup
		if jobStatus.CompletionTime == nil {
			now := metav1.Now()
			jobStatus.CompletionTime = &now
		}

		// If the Job exceeds backoff limit or is past active deadline
		// delete all pods and services, then set the status to failed
		if err := r.DeletePodsAndServices(runPolicy, job, pods); err != nil {
			return err
		}

		if err := r.CleanupJob(runPolicy, jobStatus, job); err != nil {
			return err
		}

		if r.Config.EnableGangScheduling {
			r.Recorder.Event(runtimeObject, v1.EventTypeNormal, "JobTerminated", "Job has been terminated. Deleting PodGroup")
			if err := r.DeletePodGroup(metaObject); err != nil {
				r.Recorder.Eventf(runtimeObject, v1.EventTypeWarning, "FailedDeletePodGroup", "Error deleting: %v", err)
				return err
			} else {
				r.Recorder.Eventf(runtimeObject, v1.EventTypeNormal, "SuccessfulDeletePodGroup", "Deleted PodGroup: %v", jobName)
			}
		}

		r.Recorder.Event(runtimeObject, v1.EventTypeNormal, commonutil.JobFailedReason, failureMessage)

		if err := commonutil.UpdateJobConditions(&jobStatus, commonv1.JobFailed, commonutil.JobFailedReason, failureMessage); err != nil {
			logrus.Infof("Append job condition error: %v", err)
			return err
		}

		return r.Controller.UpdateJobStatusInApiServer(job, &jobStatus)
	} else {
		// General cases which need to reconcile
		if r.Config.EnableGangScheduling {
			minMember := totalReplicas
			queue := ""
			priorityClass := ""
			var minResources *v1.ResourceList

			if runPolicy.SchedulingPolicy != nil {
				if runPolicy.SchedulingPolicy.MinAvailable != nil {
					minMember = *runPolicy.SchedulingPolicy.MinAvailable
				}

				if runPolicy.SchedulingPolicy.Queue != "" {
					queue = runPolicy.SchedulingPolicy.Queue
				}

				if runPolicy.SchedulingPolicy.PriorityClass != "" {
					priorityClass = runPolicy.SchedulingPolicy.PriorityClass
				}

				if runPolicy.SchedulingPolicy.MinResources != nil {
					minResources = runPolicy.SchedulingPolicy.MinResources
				}
			}

			if minResources == nil {
				minResources = r.calcPGMinResources(minMember, replicas)
			}

			pgSpec := v1beta1.PodGroupSpec{
				MinMember:         minMember,
				Queue:             queue,
				PriorityClassName: priorityClass,
				MinResources:      minResources,
			}

			_, err := r.SyncPodGroup(metaObject, pgSpec)
			if err != nil {
				logrus.Warnf("Sync PodGroup %v: %v", jobKey, err)
			}
		}
		ctx := context.WithValue(context.Background(), util.ContextHostNetworkPorts, make(map[string]int32))
		// Diff current active pods/services with replicas.
		// for first loop generate host port if need
		for rtype, spec := range replicas {

			err = r.ReconcileServicesCustom(ctx, metaObject, services, rtype, spec)

			if err != nil {
				logrus.Warnf("ReconcileServices error %v", err)
				return err
			}
		}

		for rtype, spec := range replicas {
			err := r.ReconcilePodsCustom(ctx, metaObject, &jobStatus, pods, rtype, spec, replicas)
			if err != nil {
				logrus.Warnf("ReconcilePods error %v", err)
				return err
			}
		}
	}

	err = r.Controller.UpdateJobStatus(job, replicas, &jobStatus)
	if err != nil {
		logrus.Warnf("UpdateJobStatus error %v", err)
		return err
	}
	// No need to update the job status if the status hasn't changed since last time.
	if !reflect.DeepEqual(*oldStatus, jobStatus) {
		return r.Controller.UpdateJobStatusInApiServer(job, &jobStatus)
	}
	return nil
}

// reconcileServices checks and updates services for each given ReplicaSpec.
// It will requeue the job in case of an error while creating/deleting services.
func (r *TFJobReconciler) ReconcileServicesCustom(
	ctx context.Context,
	job metav1.Object,
	services []*v1.Service,
	rtype commonv1.ReplicaType,
	spec *commonv1.ReplicaSpec) error {

	// Convert ReplicaType to lower string.
	rt := strings.ToLower(string(rtype))
	replicas := int(*spec.Replicas)
	// Get all services for the type rt.
	services, err := r.FilterServicesForReplicaType(services, rt)
	if err != nil {
		return err
	}

	// GetServiceSlices will return enough information here to make decision to add/remove/update resources.
	//
	// For example, let's assume we have services with replica-index 0, 1, 2
	// If replica is 4, return a slice with size 4. [[0],[1],[2],[]], a svc with replica-index 3 will be created.
	//
	// If replica is 1, return a slice with size 3. [[0],[1],[2]], svc with replica-index 1 and 2 are out of range and will be deleted.
	serviceSlices := r.GetServiceSlices(services, replicas, commonutil.LoggerForReplica(job, rt))

	for index, serviceSlice := range serviceSlices {
		if len(serviceSlice) > 1 {
			commonutil.LoggerForReplica(job, rt).Warningf("We have too many services for %s %d", rt, index)
		} else if len(serviceSlice) == 0 {
			commonutil.LoggerForReplica(job, rt).Infof("need to create new service: %s-%d", rt, index)
			err = r.CreateNewServiceCustom(ctx, job, rtype, spec, strconv.Itoa(index))
			if err != nil {
				return err
			}
		} else {
			// Check the status of the current svc.
			svc := serviceSlice[0]

			hostPort, ok := util.GetHostNetworkPortFromContext(ctx, rt, strconv.Itoa(index))
			if ok && len(svc.Spec.Ports) > 0 && svc.Spec.Ports[0].TargetPort.IntVal != hostPort {
				commonutil.LoggerForReplica(job, rt).Infof("update target service: %s-%d, new port: %d",
					rt, index, hostPort)
				// Update service target port to latest container host port, because replicas may fail-over
				// and its host port changed, so we'd ensure that other replicas can reach it with correct
				// target port.
				newService := svc.DeepCopy()
				newService.Spec.Ports[0].TargetPort = intstr.FromInt(int(hostPort))
				err = r.patchService(svc, newService)
				if err != nil {
					return err
				}
			}
			// check if the index is in the valid range, if not, we should kill the svc
			if index < 0 || index >= replicas {
				err = r.ServiceControl.DeleteService(svc.Namespace, svc.Name, job.(runtime.Object))
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (r *TFJobReconciler) calcPGMinResources(minMember int32, replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec) *v1.ResourceList {
	var replicasPriority ReplicasPriority
	for t, replica := range replicas {
		rp := ReplicaPriority{0, *replica}
		pc := replica.Template.Spec.PriorityClassName

		priorityClass, err := r.PriorityClassLister.Get(pc)
		if err != nil || priorityClass == nil {
			logrus.Warnf("Ignore task %s priority class %s: %v", t, pc, err)
		} else {
			rp.priority = priorityClass.Value
		}

		replicasPriority = append(replicasPriority, rp)
	}

	sort.Sort(replicasPriority)

	minAvailableTasksRes := v1.ResourceList{}
	podCnt := int32(0)
	for _, task := range replicasPriority {
		if task.Replicas == nil {
			continue
		}

		for i := int32(0); i < *task.Replicas; i++ {
			if podCnt >= minMember {
				break
			}
			podCnt++
			for _, c := range task.Template.Spec.Containers {
				AddResourceList(minAvailableTasksRes, c.Resources.Requests, c.Resources.Limits)
			}
		}
	}

	return &minAvailableTasksRes
}

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *TFJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)
	logger := r.Log.WithValues(tensorflowv1.Singular, req.NamespacedName)

	tfjob := &tensorflowv1.TFJob{}
	err := r.Get(ctx, req.NamespacedName, tfjob)
	if err != nil {
		logger.Info(err.Error(), "unable to fetch TFJob", req.NamespacedName.String())
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if err = validation.ValidateV1TFJobSpec(&tfjob.Spec); err != nil {
		logger.Info(err.Error(), "TFJob failed validation", req.NamespacedName.String())
	}

	// Check if reconciliation is needed
	jobKey, err := common.KeyFunc(tfjob)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get jobKey for job object %#v: %v", tfjob, err))
	}

	replicaTypes := util.GetReplicaTypes(tfjob.Spec.TFReplicaSpecs)
	needReconcile := util.SatisfiedExpectations(r.Expectations, jobKey, replicaTypes)

	if !needReconcile || tfjob.GetDeletionTimestamp() != nil {
		logger.Info("reconcile cancelled, job does not need to do reconcile or has been deleted",
			"sync", needReconcile, "deleted", tfjob.GetDeletionTimestamp() != nil)
		return ctrl.Result{}, nil
	}

	// Set default priorities to tfjob
	r.Scheme.Default(tfjob)

	// Use common to reconcile the job related pod and service
	err = r.ReconcileJobs(tfjob, tfjob.Spec.TFReplicaSpecs, tfjob.Status, &tfjob.Spec.RunPolicy)
	if err != nil {
		logrus.Warnf("Reconcile Tensorflow Job error %v", err)
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *TFJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	c, err := controller.New(r.ControllerName(), mgr, controller.Options{
		Reconciler: r,
	})

	if err != nil {
		return err
	}

	// using onOwnerCreateFunc is easier to set defaults
	if err = c.Watch(&source.Kind{Type: &tfv1.TFJob{}}, &handler.EnqueueRequestForObject{},
		predicate.Funcs{CreateFunc: r.onOwnerCreateFunc()},
	); err != nil {
		return err
	}

	// inject watching for job related pod
	if err = c.Watch(&source.Kind{Type: &corev1.Pod{}}, &handler.EnqueueRequestForOwner{
		IsController: true,
		OwnerType:    &tfv1.TFJob{},
	}, predicate.Funcs{
		CreateFunc: util.OnDependentCreateFunc(r.Expectations),
		UpdateFunc: util.OnDependentUpdateFunc(&r.JobController),
		DeleteFunc: util.OnDependentDeleteFunc(r.Expectations),
	}); err != nil {
		return err
	}

	// inject watching for job related service
	if err = c.Watch(&source.Kind{Type: &corev1.Service{}}, &handler.EnqueueRequestForOwner{
		IsController: true,
		OwnerType:    &tfv1.TFJob{},
	}, predicate.Funcs{
		CreateFunc: util.OnDependentCreateFunc(r.Expectations),
		UpdateFunc: util.OnDependentUpdateFunc(&r.JobController),
		DeleteFunc: util.OnDependentDeleteFunc(r.Expectations),
	}); err != nil {
		return err
	}

	return nil
}

func (r *TFJobReconciler) ControllerName() string {
	return controllerName
}

func (r *TFJobReconciler) GetAPIGroupVersionKind() schema.GroupVersionKind {
	return tensorflowv1.GroupVersion.WithKind(tensorflowv1.Kind)
}

func (r *TFJobReconciler) GetAPIGroupVersion() schema.GroupVersion {
	return tensorflowv1.GroupVersion
}

func (r *TFJobReconciler) GetGroupNameLabelValue() string {
	return tensorflowv1.GroupVersion.Group
}

func (r *TFJobReconciler) GetJobFromInformerCache(namespace, name string) (metav1.Object, error) {
	tfjob := &tensorflowv1.TFJob{}
	err := r.Get(context.Background(), types.NamespacedName{
		Namespace: namespace, Name: name,
	}, tfjob)
	return tfjob, err
}

func (r *TFJobReconciler) GetJobFromAPIClient(namespace, name string) (metav1.Object, error) {
	job := &tensorflowv1.TFJob{}

	clientReader, err := util.GetDelegatingClientFromClient(r.Client)
	if err != nil {
		return nil, err
	}
	err = clientReader.Get(context.Background(), types.NamespacedName{Namespace: namespace, Name: name}, job)
	if err != nil {
		if errors.IsNotFound(err) {
			logrus.Error(err, "tensorflow job not found", "namespace", namespace, "name", name)
		} else {
			logrus.Error(err, "failed to get job from api-server", "namespace", namespace, "name", name)
		}
		return nil, err
	}
	return job, nil
}

// GetPodsForJob returns the set of pods that this job should manage.
// It also reconciles ControllerRef by adopting/orphaning.
// Note that the returned Pods are pointers into the cache.
func (r *TFJobReconciler) GetPodsForJob(jobObject interface{}) ([]*corev1.Pod, error) {
	job, ok := jobObject.(metav1.Object)
	if !ok {
		return nil, fmt.Errorf("job is not of type metav1.Object")
	}

	// Create selector.
	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{
		MatchLabels: r.GenLabels(job.GetName()),
	})

	if err != nil {
		return nil, fmt.Errorf("couldn't convert Job selector: %v", err)
	}
	// List all pods to include those that don't match the selector anymore
	// but have a ControllerRef pointing to this controller.
	podlist := &corev1.PodList{}
	err = r.List(context.Background(), podlist,
		client.MatchingLabelsSelector{Selector: selector}, client.InNamespace(job.GetNamespace()))
	if err != nil {
		return nil, err
	}

	pods := util.ConvertPodList(podlist.Items)

	// If any adoptions are attempted, we should first recheck for deletion
	// with an uncached quorum read sometime after listing Pods (see #42639).
	canAdoptFunc := common.RecheckDeletionTimestamp(func() (metav1.Object, error) {
		fresh, err := r.Controller.GetJobFromAPIClient(job.GetNamespace(), job.GetName())
		if err != nil {
			return nil, err
		}
		if fresh.GetUID() != job.GetUID() {
			return nil, fmt.Errorf("original Job %v/%v is gone: got uid %v, wanted %v", job.GetNamespace(), job.GetName(), fresh.GetUID(), job.GetUID())
		}
		return fresh, nil
	})
	cm := control.NewPodControllerRefManager(r.PodControl, job, selector, r.Controller.GetAPIGroupVersionKind(), canAdoptFunc)
	return cm.ClaimPods(pods)
}

// GetServicesForJob returns the set of services that this job should manage.
// It also reconciles ControllerRef by adopting/orphaning.
// Note that the returned services are pointers into the cache.
func (r *TFJobReconciler) GetServicesForJob(jobObject interface{}) ([]*corev1.Service, error) {
	job, ok := jobObject.(metav1.Object)
	if !ok {
		return nil, fmt.Errorf("job is not of type metav1.Object")
	}

	// Create selector
	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{
		MatchLabels: r.GenLabels(job.GetName()),
	})

	if err != nil {
		return nil, fmt.Errorf("couldn't convert Job selector: %v", err)
	}
	// List all services to include those that don't match the selector anymore
	// but have a ControllerRef pointing to this controller.
	svclist := &corev1.ServiceList{}
	err = r.List(context.Background(), svclist,
		client.MatchingLabelsSelector{Selector: selector}, client.InNamespace(job.GetNamespace()))
	if err != nil {
		return nil, fmt.Errorf("couldn't get Service: %v", err)
	}

	// If any adoptions are attempted, we should first recheck for deletion
	// with an uncached quorum read sometime after listing services (see #42639).
	canAdoptFunc := common.RecheckDeletionTimestamp(func() (metav1.Object, error) {
		fresh, err := r.GetJobFromInformerCache(job.GetNamespace(), job.GetName())
		if err != nil {
			return nil, err
		}
		if fresh.GetUID() != job.GetUID() {
			return nil, fmt.Errorf("original Job %v/%v is gone: got uid %v, wanted %v", job.GetNamespace(), job.GetName(), fresh.GetUID(), job.GetUID())
		}
		return fresh, nil
	})
	cm := control.NewServiceControllerRefManager(r.ServiceControl, job, selector, r.Controller.GetAPIGroupVersionKind(), canAdoptFunc)

	services := util.ConvertServiceList(svclist.Items)
	return cm.ClaimServices(services)
}

func (r *TFJobReconciler) DeleteJob(job interface{}) error {
	tfJob, ok := job.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfJob)
	}

	log := commonutil.LoggerForJob(tfJob)
	if err := r.Delete(context.Background(), tfJob); err != nil {
		r.recorder.Eventf(tfJob, v1.EventTypeWarning, FailedDeleteJobReason, "Error deleting: %v", err)
		log.Errorf("failed to delete job %s/%s, %v", tfJob.Namespace, tfJob.Name, err)
		return err
	}

	r.recorder.Eventf(tfJob, v1.EventTypeNormal, SuccessfulDeleteJobReason, "Deleted job: %v", tfJob.Name)
	logrus.Infof("job %s/%s has been deleted", tfJob.Namespace, tfJob.Name)
	trainingoperatorcommon.DeletedJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
	return nil
}

func (r *TFJobReconciler) UpdateJobStatus(job interface{}, replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec, jobStatus *commonv1.JobStatus) error {
	tfJob, ok := job.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfJob)
	}

	tfJobKey, err := common.KeyFunc(tfJob)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for tfjob object %#v: %v", tfJob, err))
		return err
	}

	logger := commonutil.LoggerForJob(tfJob)

	worker0Completed, err := r.IsWorker0Completed(tfJob, replicas)
	if err != nil {
		logger.Warnf("check if worker 0 completed error %v", err)
		return err
	}

	// Set StartTime.
	if jobStatus.StartTime == nil {
		now := metav1.Now()
		jobStatus.StartTime = &now
		// enqueue a sync to check if job past ActiveDeadlineSeconds
		if tfJob.Spec.RunPolicy.ActiveDeadlineSeconds != nil {
			logger.Infof("Job with ActiveDeadlineSeconds will sync after %d seconds", *tfJob.Spec.RunPolicy.ActiveDeadlineSeconds)
			// TODO(Jeffwan): requeue job key in reconciler scenarios
			r.WorkQueue.AddAfter(tfJobKey, time.Duration(*tfJob.Spec.RunPolicy.ActiveDeadlineSeconds)*time.Second)
		}
	}
	// iterate the replica spec based on this order
	allTypes := []commonv1.ReplicaType{
		tensorflowv1.TFReplicaTypeChief,
		tensorflowv1.TFReplicaTypeEval,
		tensorflowv1.TFReplicaTypeMaster,
		tensorflowv1.TFReplicaTypePS,
		tensorflowv1.TFReplicaTypeWorker,
	}
	for _, rtype := range allTypes {
		if replicas[rtype] == nil {
			continue
		}
		spec := replicas[rtype]
		status := jobStatus.ReplicaStatuses[rtype]

		// Expect to have `replicas - succeeded` pods alive.
		succeeded := status.Succeeded
		expected := *(spec.Replicas) - succeeded
		running := status.Active
		failed := status.Failed

		logger.Infof("TFJob=%s/%s, ReplicaType=%s expected=%d, running=%d, failed=%d",
			tfJob.Namespace, tfJob.Name, rtype, expected, running, failed)

		// If the TFJob contains Chief or Master spec, then we will update the status
		// according to the Chief/Master spec.
		if ContainsChiefOrMasterSpec(tfJob.Spec.TFReplicaSpecs) {
			if tensorflowv1.IsChieforMaster(rtype) {
				if running > 0 {
					msg := fmt.Sprintf("TFJob %s/%s is running.",
						tfJob.Namespace, tfJob.Name)
					err := commonutil.UpdateJobConditions(jobStatus,
						commonv1.JobRunning, tfJobRunningReason, msg)
					if err != nil {
						commonutil.LoggerForJob(tfJob).Infof(
							"Append tfjob condition error: %v", err)
						return err
					}
				}
				if expected == 0 {
					msg := fmt.Sprintf("TFJob %s/%s successfully completed.",
						tfJob.Namespace, tfJob.Name)
					r.recorder.Event(tfJob, corev1.EventTypeNormal, tfJobSucceededReason, msg)
					if jobStatus.CompletionTime == nil {
						now := metav1.Now()
						jobStatus.CompletionTime = &now
					}
					err := commonutil.UpdateJobConditions(jobStatus,
						commonv1.JobSucceeded, tfJobSucceededReason, msg)
					if err != nil {
						commonutil.LoggerForJob(tfJob).Infof("Append tfjob condition error: %v", err)
						return err
					}
					trainingoperatorcommon.SuccessfulJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
				}
			}
		} else {
			if rtype == tensorflowv1.TFReplicaTypeWorker {
				// Leave a succeeded condition for the following two cases:
				// 1. If default success policy is used and worker 0 has completed.
				// 2. If `SuccessPolicyAllWorkers` success policy is used and all workers are succeeded.
				if expected == 0 || (worker0Completed && *tfJob.Spec.SuccessPolicy != tensorflowv1.SuccessPolicyAllWorkers) {
					msg := fmt.Sprintf("TFJob %s/%s successfully completed.",
						tfJob.Namespace, tfJob.Name)
					r.recorder.Event(tfJob, corev1.EventTypeNormal, tfJobSucceededReason, msg)
					if jobStatus.CompletionTime == nil {
						now := metav1.Now()
						jobStatus.CompletionTime = &now
					}
					err := commonutil.UpdateJobConditions(jobStatus,
						commonv1.JobSucceeded, tfJobSucceededReason, msg)
					if err != nil {
						commonutil.LoggerForJob(tfJob).Infof("Append tfjob condition error: %v", err)
						return err
					}
					trainingoperatorcommon.SuccessfulJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
				} else if running > 0 {
					// Some workers are still running, leave a running condition.
					msg := fmt.Sprintf("TFJob %s/%s is running.",
						tfJob.Namespace, tfJob.Name)
					err := commonutil.UpdateJobConditions(jobStatus, commonv1.JobRunning, tfJobRunningReason, msg)
					if err != nil {
						commonutil.LoggerForJob(tfJob).Infof("Append tfjob condition error: %v", err)
						return err
					}
				}
			}
		}

		if failed > 0 {
			restart := false
			for _, condition := range jobStatus.Conditions {
				if condition.Type == commonv1.JobRestarting {
					restart = true
				}
			}

			if restart {
				// job is restarting, no need to set it failed
				// we know it because we update the status condition when reconciling the replicas
				trainingoperatorcommon.RestartedJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
			} else {
				msg := fmt.Sprintf("TFJob %s/%s has failed because %d %s replica(s) failed.",
					tfJob.Namespace, tfJob.Name, failed, rtype)
				r.recorder.Event(tfJob, corev1.EventTypeNormal, tfJobFailedReason, msg)
				if jobStatus.CompletionTime == nil {
					now := metav1.Now()
					jobStatus.CompletionTime = &now
				}
				err := commonutil.UpdateJobConditions(jobStatus,
					commonv1.JobFailed, tfJobFailedReason, msg)
				if err != nil {
					commonutil.LoggerForJob(tfJob).Infof("Append tfjob condition error: %v", err)
					return err
				}
				trainingoperatorcommon.FailedJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
			}
		}
	}
	// we assign the jobStatus to the tfJob.Status for testing purpose
	// it won't effect the main reconcile logic
	// because we already use oldStatus := jobStatus.DeepCopy() to record the oldStatus
	// and use !reflect.DeepEqual(*oldStatus, jobStatus) to decide whether to update the tfJob or not
	tfJob.Status = *jobStatus.DeepCopy()

	return nil
}

func (r *TFJobReconciler) UpdateJobStatusInApiServer(job interface{}, jobStatus *commonv1.JobStatus) error {
	tfJob, ok := job.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfJob)
	}

	startTime := time.Now()
	logger := commonutil.LoggerForJob(tfJob)
	defer func() {
		logger.Infof("Finished updating TFJobs Status %q (%v)",
			tfJob.Name, time.Since(startTime))
	}()

	tfJob = tfJob.DeepCopy()
	tfJob.Status = *jobStatus.DeepCopy()

	result := r.Status().Update(context.Background(), tfJob)

	if result != nil {
		r.Log.WithValues("tfjob", types.NamespacedName{
			Namespace: tfJob.GetNamespace(),
			Name:      tfJob.GetName(),
		})
		return result
	}

	return nil
}

// Same as Func (tc *TFController) SetClusterSpec(...) in pod.go
func (r *TFJobReconciler) SetClusterSpecCustom(ctx context.Context, job interface{}, podTemplate *corev1.PodTemplateSpec, rtype, index string) error {
	tfjob, ok := job.(*tensorflowv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfjob)
	}

	// Do not set TF_CONFIG for local training jobs.
	if !isDistributed(tfjob) {
		return nil
	}
	// Generate TF_CONFIG JSON string.
	tfConfigStr, err := genTFConfigJSONStr(ctx, tfjob, rtype, index)
	if err != nil {
		return err
	}

	if tfConfigStr == "" {
		return nil
	}
	// Add TF_CONFIG environment variable to tensorflow container in the pod.
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == tensorflowv1.DefaultContainerName {
			if len(podTemplate.Spec.Containers[i].Env) == 0 {
				podTemplate.Spec.Containers[i].Env = make([]corev1.EnvVar, 0)
			}
			podTemplate.Spec.Containers[i].Env = append(podTemplate.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  tfConfig,
				Value: tfConfigStr,
			})
			break
		}
	}
	return nil
}

func (r *TFJobReconciler) SetClusterSpec(job interface{}, podTemplate *corev1.PodTemplateSpec, rtype, index string) error {
	return nil
}

func (r *TFJobReconciler) GetDefaultContainerName() string {
	return tensorflowv1.DefaultContainerName
}

func (r *TFJobReconciler) GetDefaultContainerPortName() string {
	return tensorflowv1.DefaultPortName
}

func (r *TFJobReconciler) IsMasterRole(replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec,
	rtype commonv1.ReplicaType, index int) bool {
	if ContainsChiefOrMasterSpec(replicas) {
		return rtype == tensorflowv1.TFReplicaTypeChief || rtype == tensorflowv1.TFReplicaTypeMaster
	}
	// else check if it is worker with index 0
	return rtype == tensorflowv1.TFReplicaTypeWorker && index == 0
}

// IsWorker0Completed returns true if pod of worker0 succeeded and exited with 0
func (r *TFJobReconciler) IsWorker0Completed(tfjob *tensorflowv1.TFJob, replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec) (bool, error) {
	worker0Completed := false
	_, ok := replicas[tensorflowv1.TFReplicaTypeWorker]
	if !ok {
		return true, nil
	}
	podSlices, err := r.getPodSlices(tfjob, replicas[tensorflowv1.TFReplicaTypeWorker].Replicas)
	if err != nil {
		return false, err
	}
	for index, podSlice := range podSlices {
		if len(podSlice) == 1 {
			pod := podSlice[0]
			exitCode := getContainerExitCode(pod)
			if index == 0 && exitCode == 0 && pod.Status.Phase == v1.PodSucceeded {
				worker0Completed = true
			}
		}
	}
	return worker0Completed, nil
}

// getPodSlices returns a slice, which element is the slice of pod.
// It gives enough information to caller to make decision to up/down scale resources.
func (r *TFJobReconciler) getPodSlices(tfjob *tensorflowv1.TFJob, replicasNum *int32) ([][]*v1.Pod, error) {
	logger := commonutil.LoggerForReplica(tfjob, strings.ToLower(string(tensorflowv1.TFReplicaTypeWorker)))

	pods, err := r.GetPodsForJob(tfjob)
	if err != nil {
		commonutil.LoggerForJob(tfjob).Warnf("getPodsForTFJob error %v", err)
		return nil, err
	}

	// Get all pods for the type rt.
	pods, err = r.JobController.FilterPodsForReplicaType(pods, strings.ToLower(string(tensorflowv1.TFReplicaTypeWorker)))
	if err != nil {
		return nil, err
	}

	podSlices := r.GetPodSlices(pods, int(*replicasNum), logger)
	return podSlices, nil
}

// In order to minimize the changes, we copy TFController's logic here to override kubeflow/commons reconcile logic
// This should be removed later unless TF has specific logics there
// reconcilePods checks and updates pods for each given TFReplicaSpec.
// It will requeue the tfjob in case of an error while creating/deleting pods.
func (r *TFJobReconciler) ReconcilePodsCustom(
	ctx context.Context,
	job interface{},
	jobStatus *commonv1.JobStatus,
	pods []*v1.Pod,
	rtype commonv1.ReplicaType,
	spec *commonv1.ReplicaSpec,
	replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec,
) error {

	tfJob, ok := job.(*tfv1.TFJob)
	if !ok {
		return fmt.Errorf("%v is not a type of TFJob", tfJob)
	}

	// Convert ReplicaType to lower string.
	rt := strings.ToLower(string(rtype))
	logger := commonutil.LoggerForJob(tfJob)
	// Get all pods for the type rt.
	pods, err := r.FilterPodsForReplicaType(pods, rt)
	if err != nil {
		return err
	}
	numReplicas := int(*spec.Replicas)
	masterRole := false
	//restart := false
	//worker0Completed := false

	initializeReplicaStatuses(jobStatus, rtype)

	// GetPodSlices will return enough information here to make decision to add/remove/update resources.
	//
	// For example, let's assume we have pods with replica-index 0, 1, 2
	// If replica is 4, return a slice with size 4. [[0],[1],[2],[]], a pod with replica-index 3 will be created.
	//
	// If replica is 1, return a slice with size 3. [[0],[1],[2]], pod with replica-index 1 and 2 are out of range and will be deleted.
	podSlices := r.GetPodSlices(pods, numReplicas, logger)
	for index, podSlice := range podSlices {
		if len(podSlice) > 1 {
			logger.Warningf("We have too many pods for %s %d", rt, index)
		} else if len(podSlice) == 0 {
			logger.Infof("Need to create new pod: %s-%d", rt, index)

			// check if this replica is the master role
			masterRole = r.IsMasterRole(replicas, rtype, index)
			// TODO: [should change to CreateNewPod]
			err = r.createNewPod(ctx, tfJob, rt, strconv.Itoa(index), spec, masterRole, replicas)
			if err != nil {
				return err
			}
		} else {
			// Check the status of the current pod.
			pod := podSlice[0]

			// check if the index is in the valid range, if not, we should kill the pod
			if index < 0 || index >= numReplicas {
				err = r.PodControl.DeletePod(pod.Namespace, pod.Name, tfJob)
				if err != nil {
					return err
				}
			}
			// Get the exit code of the container.
			var exitCode int32 = 0xbeef // magic number
			for _, status := range pod.Status.ContainerStatuses {
				state := status.State
				if status.Name == r.GetDefaultContainerName() && state.Terminated != nil {
					exitCode = state.Terminated.ExitCode
					logger.Infof("Pod: %v.%v exited with code %v", pod.Namespace, pod.Name, exitCode)
					r.Recorder.Eventf(tfJob, v1.EventTypeNormal, exitedWithCodeReason, "Pod: %v.%v exited with code %v", pod.Namespace, pod.Name, exitCode)
				}
			}
			// Get and pass its container port by context if pod enables hostnetwork mode.
			if util.EnableHostNetwork(tfJob) {
				port := util.GetContainerHostNetworkPort(pod, r.Controller.GetDefaultContainerName(), r.Controller.GetDefaultContainerPortName())
				logger.Debugf("## HostPort Set to Context: %v.%v ", pod.Name, port)
				util.StoreHostNetworkPortToContext(ctx, rt, strconv.Itoa(index), port)
			}

			// Check if the pod is retryable.
			if spec.RestartPolicy == commonv1.RestartPolicyExitCode {
				if pod.Status.Phase == v1.PodFailed && train_util.IsRetryableExitCode(exitCode) {
					logger.Infof("Need to restart the pod: %v.%v", pod.Namespace, pod.Name)
					if err := r.PodControl.DeletePod(pod.Namespace, pod.Name, tfJob); err != nil {
						return err
					}

					// with common library framework, we have to handle restart status here
					// or we won't know which replica has been restarted in updateJobStatus after reconciling all replicas
					msg := fmt.Sprintf("TFJob %s is restarting because %s replica(s) failed.",
						tfJob.Name, rtype)
					r.Recorder.Event(tfJob, corev1.EventTypeWarning, tfJobRestartingReason, msg)
					err := commonutil.UpdateJobConditions(jobStatus, commonv1.JobRestarting, tfJobRestartingReason, msg)
					if err != nil {
						commonutil.LoggerForJob(tfJob).Infof("Append tfjob condition error: %v", err)
						return err
					}
					trainingoperatorcommon.RestartedJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
				}
			}

			updateJobReplicaStatuses(jobStatus, rtype, pod)
		}
	}
	return nil
}

// createNewPod creates a new pod for the given index and type.
func (r *TFJobReconciler) createNewPod(ctx context.Context, tfjob *tfv1.TFJob, rt, index string, spec *commonv1.ReplicaSpec, masterRole bool,
	replicas map[commonv1.ReplicaType]*commonv1.ReplicaSpec) error {

	tfjobKey, err := common.KeyFunc(tfjob)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for tfjob object %#v: %v", tfjob, err))
		return err
	}
	expectationPodsKey := expectation.GenExpectationPodsKey(tfjobKey, rt)
	err = r.Expectations.ExpectCreations(expectationPodsKey, 1)
	if err != nil {
		return err
	}
	logger := commonutil.LoggerForReplica(tfjob, rt)
	// Create OwnerReference.
	controllerRef := r.GenOwnerReference(tfjob)

	// Set type and index for the worker.
	labels := r.GenLabels(tfjob.Name)
	labels[tfReplicaTypeLabel] = rt
	labels[tfReplicaIndexLabel] = index
	podTemplate := spec.Template.DeepCopy()

	if masterRole {
		labels[commonv1.JobRoleLabel] = "master"
	}
	if util.EnableHostNetwork(tfjob) {
		commonutil.LoggerForReplica(tfjob, rt).Infof("pod enable host network, name: %s, masterRole: %v",
			tfjob.GetName(), masterRole)
		if err := r.setupHostNetwork(ctx, podTemplate, rt, index); err != nil {
			return err
		}

	}

	// Set name for the template.
	podTemplate.Name = common.GenGeneralName(tfjob.Name, rt, index)

	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}

	for key, value := range labels {
		podTemplate.Labels[key] = value
	}

	if err := r.SetClusterSpecCustom(ctx, tfjob, podTemplate, rt, index); err != nil {
		return err
	}

	// Submit a warning event if the user specifies restart policy for
	// the pod template. We recommend to set it from the replica level.
	if podTemplate.Spec.RestartPolicy != v1.RestartPolicy("") {
		errMsg := "Restart policy in pod template will be overwritten by restart policy in replica spec"
		logger.Warning(errMsg)
		r.Recorder.Event(tfjob, v1.EventTypeWarning, podTemplateRestartPolicyReason, errMsg)
	}
	setRestartPolicy(podTemplate, spec)

	// if gang-scheduling is enabled:
	// 1. if user has specified other scheduler, we report a warning without overriding any fields.
	// 2. if no SchedulerName is set for pods, then we set the SchedulerName to "volcano".
	if r.Config.EnableGangScheduling {
		podSchedulerName := util.GetSchedulerName(replicas)
		if len(podSchedulerName) == 0 {
			podTemplate.Spec.SchedulerName = gangSchedulerName
		} else if strings.Compare(podSchedulerName, gangSchedulerName) != 0 {
			errMsg := "Another scheduler is specified when gang-scheduling is enabled and it will not be overwritten"
			logger.Warning(errMsg)
			r.Recorder.Event(tfjob, v1.EventTypeWarning, podTemplateSchedulerNameReason, errMsg)
		}

		if podTemplate.Annotations == nil {
			podTemplate.Annotations = map[string]string{}
		}
		podTemplate.Annotations[gangSchedulingPodGroupAnnotation] = tfjob.GetName()
		podTemplate.Annotations[volcanoTaskSpecKey] = rt
	}

	err = r.PodControl.CreatePodsWithControllerRef(tfjob.Namespace, podTemplate, tfjob, controllerRef)
	if err != nil && errors.IsTimeout(err) {
		// Pod is created but its initialization has timed out.
		// If the initialization is successful eventually, the
		// controller will observe the creation via the informer.
		// If the initialization fails, or if the pod keeps
		// uninitialized for a long time, the informer will not
		// receive any update, and the controller will create a new
		// pod when the expectation expires.
		return nil
	} else if err != nil {
		// Decrement the expected number of creates because the informer won't observe this pod
		logger.Infof(
			"Failed creation, decrementing expectations for tfjob %s/%s, key %s",
			tfjob.Namespace, tfjob.Name, expectationPodsKey)
		r.Expectations.CreationObserved(expectationPodsKey)
		return err
	}
	return nil
}

func (r *TFJobReconciler) setupHostNetworkOrigin(ctx context.Context, spec *v1.PodTemplateSpec, rtype, index string) error {
	const (
		randomPortLowerBound = 20001
		randomPortUpperBound = 65535
	)
	// 先设置第一个节点
	port := int32(rand.IntnRange(randomPortLowerBound, randomPortUpperBound))
	// 1) enable pod hostNetwork mode.
	spec.Spec.HostNetwork = true
	// 2) [CRITICAL] setup dns policy with hostnetwork instead of ClusterFirst by default.
	spec.Spec.DNSPolicy = v1.DNSClusterFirstWithHostNet
	// 3) setup container port with a random port ranged [20001, 65535).
	util.SetupContainerHostNetworkPort(spec, r.Controller.GetDefaultContainerName(), r.Controller.GetDefaultContainerPortName(), port)
	// 4) record selected port by context keyed with replica-index.
	util.StoreHostNetworkPortToContext(ctx, rtype, index, port)

	return nil
}

func (r *TFJobReconciler) setupHostNetwork(ctx context.Context, spec *v1.PodTemplateSpec, rtype, index string) error {

	// 先设置第一个节点
	port, ok := util.GetHostNetworkPortFromContext(ctx, rtype, index)
	if !ok {
		logrus.Warnf("Can't get hostport from context: %v-%v", rtype, index)
		return nil
	}
	// 1) enable pod hostNetwork mode.
	spec.Spec.HostNetwork = true
	// 2) [CRITICAL] setup dns policy with hostnetwork instead of ClusterFirst by default.
	spec.Spec.DNSPolicy = v1.DNSClusterFirstWithHostNet
	// 3) setup container port with a random port ranged [20001, 65535).
	util.SetupContainerHostNetworkPort(spec, r.Controller.GetDefaultContainerName(), r.Controller.GetDefaultContainerPortName(), port)
	// 4) record selected port by context keyed with replica-index.
	util.StoreHostNetworkPortToContext(ctx, rtype, index, port)

	return nil
}

// onOwnerCreateFunc modify creation condition.
func (r *TFJobReconciler) onOwnerCreateFunc() func(event.CreateEvent) bool {
	return func(e event.CreateEvent) bool {
		tfJob, ok := e.Object.(*tensorflowv1.TFJob)
		if !ok {
			return true
		}

		r.Scheme.Default(tfJob)
		msg := fmt.Sprintf("TFJob %s is created.", e.Object.GetName())
		logrus.Info(msg)
		trainingoperatorcommon.CreatedJobsCounterInc(tfJob.Namespace, tensorflowv1.FrameworkName)
		if err := commonutil.UpdateJobConditions(&tfJob.Status, commonv1.JobCreated, "TFJobCreated", msg); err != nil {
			log.Log.Error(err, "append job condition error")
			return false
		}
		return true
	}
}

// createNewService creates a new service for the given index and type.
func (r *TFJobReconciler) CreateNewServiceCustom(ctx context.Context, job metav1.Object, rtype commonv1.ReplicaType,
	spec *commonv1.ReplicaSpec, index string) error {
	const (
		randomPortLowerBound = 20001
		randomPortUpperBound = 65535
	)

	jobKey, err := KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for job object %#v: %v", job, err))
		return err
	}

	// Convert ReplicaType to lower string.
	rt := strings.ToLower(string(rtype))

	// Append ReplicaTypeLabel and ReplicaIndexLabel labels.
	labels := r.GenLabels(job.GetName())
	labels[commonv1.ReplicaTypeLabel] = rt
	labels[commonv1.ReplicaIndexLabel] = index

	ports, err := r.GetPortsFromJob(spec)
	if err != nil {
		return err
	}
	clusterIP := "None"

	service := &v1.Service{
		Spec: v1.ServiceSpec{
			ClusterIP: clusterIP,
			Selector:  labels,
			Ports:     []v1.ServicePort{},
		},
	}

	// Add service ports to headless service
	for name, port := range ports {
		if name == r.GetDefaultContainerPortName() && util.EnableHostNetwork(job) {
			// 先创建svc svc的loop中生成hostPort
			newHostPort := int32(rand.IntnRange(randomPortLowerBound, randomPortUpperBound))
			logrus.Infof("genrating hostPort Here: %v", newHostPort)
			util.StoreHostNetworkPortToContext(ctx, rt, index, newHostPort)
			svcPort := v1.ServicePort{Name: name, Port: newHostPort}
			service.Spec.Ports = append(service.Spec.Ports, svcPort)

		} else {
			svcPort := v1.ServicePort{Name: name, Port: port}
			service.Spec.Ports = append(service.Spec.Ports, svcPort)

		}
	}

	service.Name = GenGeneralName(job.GetName(), rt, index)
	service.Labels = labels
	// Create OwnerReference.
	controllerRef := r.GenOwnerReference(job)

	// Creation is expected when there is no error returned
	expectationServicesKey := expectation.GenExpectationServicesKey(jobKey, rt)
	r.Expectations.RaiseExpectations(expectationServicesKey, 1, 0)

	err = r.ServiceControl.CreateServicesWithControllerRef(job.GetNamespace(), service, job.(runtime.Object), controllerRef)
	if err != nil && errors.IsTimeout(err) {
		// Service is created but its initialization has timed out.
		// If the initialization is successful eventually, the
		// controller will observe the creation via the informer.
		// If the initialization fails, or if the service keeps
		// uninitialized for a long time, the informer will not
		// receive any update, and the controller will create a new
		// service when the expectation expires.
		succeededServiceCreationCount.Inc()
		return nil
	} else if err != nil {
		// Since error occurred(the informer won't observe this service),
		// we decrement the expected number of creates
		// and wait until next reconciliation
		r.Expectations.CreationObserved(expectationServicesKey)
		failedServiceCreationCount.Inc()
		return err
	}
	succeededServiceCreationCount.Inc()
	return nil
}
func (r *TFJobReconciler) patchService(oldObj, newObj *v1.Service) error {
	// deepcopy new object avoid of in-memory modifications being override by in-cluster object.
	newPatchObj := newObj.DeepCopyObject()
	return r.Client.Patch(context.Background(), newPatchObj.(*v1.Service), client.MergeFrom(oldObj))
}
