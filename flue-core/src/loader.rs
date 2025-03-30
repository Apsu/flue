use std::future::Future;

use anyhow::Result;
use hf_hub::api::tokio::Api;

use crate::{DeviceMap, ModelLike};

pub trait Loader {
    type Model: ModelLike;

    fn load(api: Api, device_map: DeviceMap) -> impl Future<Output = Result<Self::Model>>
    where
        Self: Sized;
}
