/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"

#include "vk_log.h"
#include "util/u_sync_provider.h"

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateEvent)(VkDevice _device,
                            const VkEventCreateInfo *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            VkEvent *pEvent)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   struct panvk_event *event = vk_object_zalloc(
      &device->vk, pAllocator, sizeof(*event), VK_OBJECT_TYPE_EVENT);
   if (!event)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Through the device's sync provider: on kbase there are no kernel syncobjs. */
   int ret = device->vk.sync->create(device->vk.sync, 0, &event->syncobj);
   if (ret) {
      vk_object_free(&device->vk, pAllocator, event);
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pEvent = panvk_event_to_handle(event);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyEvent)(VkDevice _device, VkEvent _event,
                             const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   if (!event)
      return;

   device->vk.sync->destroy(device->vk.sync, event->syncobj);

   vk_object_free(&device->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetEventStatus)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);
   bool signaled;

   int ret = device->vk.sync->wait(device->vk.sync, &event->syncobj, 1, 0,
                                   DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
   if (ret) {
      if (ret == -ETIME)
         signaled = false;
      else {
         assert(0);
         return VK_ERROR_DEVICE_LOST; /* TODO */
      }
   } else
      signaled = true;

   return signaled ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(SetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   /* This is going to just replace the fence for this syncobj with one that
    * is already in signaled state. This won't be a problem because the spec
    * mandates that the event will have been set before the vkCmdWaitEvents
    * command executes.
    * https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html#commandbuffers-submission-progress
    */
   if (device->vk.sync->signal(device->vk.sync, &event->syncobj, 1))
      return VK_ERROR_DEVICE_LOST;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(ResetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_event, event, _event);

   if (device->vk.sync->reset(device->vk.sync, &event->syncobj, 1))
      return VK_ERROR_DEVICE_LOST;

   return VK_SUCCESS;
}
