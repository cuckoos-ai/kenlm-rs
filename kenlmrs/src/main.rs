

use autocxx::prelude::*; // use all the main autocxx functions

include_cpp! {
    // include "my_header.h" // your header file name
    // safety!(unsafe) // see details of unsafety policies described in the 'safety' section of the book
    // generate!("DeepThought") // add this line for each function or type you wish to generate
}
