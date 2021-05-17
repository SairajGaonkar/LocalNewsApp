import React, {useEffect} from 'react';
import { View, Text, TextInput,StyleSheet, TouchableOpacity, Button, ListView, FlatList } from 'react-native';
import AsyncStorage  from '@react-native-async-storage/async-storage';
import axios from 'axios';
import deviceStorage from "../Services/DeviceStorage.js";

// let url='http://192.168.0.146:5000';
let url='http://127.0.0.1:5000';
const ProfilePage = ({ err, navigation }) => { 
    const styles = StyleSheet.create({
        flexCol: {
            // flexDirection: "row",
            padding: 10,
          },
          title: {
            
            fontSize: 20,
            color: '#003f5c',
            marginTop: 5,
          },
          input: {
            borderStartWidth: 2,
            borderEndWidth: 2,
            borderTopWidth: 2,
            borderLeftWidth: 2,
            borderRightWidth: 2,
            borderBottomWidth: 2,
            borderRadius: 20,
            padding: 10,
            fontSize: 18,
            backgroundColor: "#fff",
            borderColor: "#aaa",
            width: 350,
            marginTop: 5,
            
          },
          Btn:{
            width: 350,
            backgroundColor:"#fb5b5a",
            borderRadius:25,
            height:50,
            alignItems:"center",
            justifyContent:"center",
            marginTop:20,
          }
    })
    const[curr_zipcode, setZipCode] = React.useState('');
    const[curr_user, setUser]=React.useState('');
    const[firstName, setFirstName] = React.useState('');
    const[lastName, setLastName] = React.useState('');
    const[zipcode, setZipCodeNo] = React.useState('');
    const[email, setEmail] = React.useState('');

    const onUpdate = (firstName, lastName, zipcode, email) => {
        if(firstName==""){
            
            setFirstName(curr_user["first_name"]);
            console.log("fffffff", firstName);
        }
        if(lastName==null){
            setLastName(curr_user["last_name"]);
        }
        if(email==null){
            setEmail(curr_user["email_id"]);
        }
        // console.log("firstName:" , firstName);
        // console.log("lastName:" , lastName);
        // console.log("zipcode:" , zipcode);
        // console.log('email:',email);
        
        //console.log('password:',password);
        // TODO implement real sign up mechanism
        let payload = {
          zip : zipcode,
          email_id : email
        };
        console.log(payload)

        axios
        .put(url+"/zipcodeChange", payload, {
          headers: {
            "content-type": "application/json",
          },
        })
        .then((response) => {
            console.log("Response is", response);
          if(response){
            console.log(response.data)
            let user = JSON.stringify(response.data);
            deviceStorage.saveKey("user_data", user);
          }
          else{
            //setRegisError(true);
          }
        })
        .catch((err) => {
          console.log(err);
          //setRegisError(true);
        });
      };
 
    useEffect(()=>{
        AsyncStorage.getItem('user_data', (err, result) => {
      
            if(result){
             
              let user =  JSON.parse(result);
              setUser(user);
              
              
              let zip_code = user['zip_code'];
              setZipCodeNo(zip_code);
              setFirstName(user['first_name']);
              setLastName(user['last_name']);
              setEmail(user['email_id']);
              //console.log("Zipcode is",zip_code);
              // console.log("firstName:" , firstName);
              // console.log("lastName:" , lastName);
              // console.log("zipcode:" , zipcode);
              // console.log('email:',email);
            }
        })
    },[]);

    return (
        
        <View style={styles.flexCol}>
        <Text style={styles.title}>First Name:  </Text>
        <TextInput
            style={styles.input}
            placeholder="First Name"
            defaultValue={curr_user["first_name"]}
            // onChange={e => setFirstName(e.nativeEvent.text)}
            // disabled = "true"
            editable={false}
          />

        <Text style={styles.title}>Last Name:  </Text>
        <TextInput
            style={styles.input}
            placeholder="Last Name"
            defaultValue={curr_user["last_name"]}
            // onChange={e => setLastName(e.nativeEvent.text)}
            editable={false}
           
          />

        <Text style={styles.title}>Email Id:  </Text>
        <TextInput
            style={styles.input}
            placeholder="Email id"
            defaultValue={curr_user["email_id"]}
            // onChange={e => setEmail(e.nativeEvent.text)}
            editable={false}
           
          />
        
        <Text style={styles.title}>Zipcode:  </Text>
        <TextInput
            style={styles.input}
            placeholder="Zipcode"
            defaultValue={curr_user["zip_code"]}
            onChange={e => setZipCodeNo(e.nativeEvent.text)}
           
          />

        <View style = {styles.Btn}>
            <TouchableOpacity  onPress = {() => onUpdate(firstName, lastName, zipcode, email)}>
            <Text style = {styles.White} >UPDATE ZIPCODE </Text>
            </TouchableOpacity>
        </View>
        </View>
      
     
    )
  };

export default ProfilePage;
