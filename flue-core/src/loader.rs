use std::future::Future;

use anyhow::Result;
use hf_hub::api::tokio::Api;
use loader_factory::ModelVariant;

use crate::{loader_factory, DeviceMap, ModelLike};

pub trait Loader {
    type Model: ModelLike;

    fn load(
        variant: ModelVariant,
        api: Api,
        device_map: DeviceMap,
    ) -> impl Future<Output = Result<Self::Model>>
    where
        Self: Sized;
}
